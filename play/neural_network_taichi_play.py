__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Python standard library imports
import logging
from typing import List
import python
import torch
import taichi as ti
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Identify the type of processor following the sequence ti.cuda, ti.vulkan, ti.metal and ti.opengl
# Fast math set to true using 32-bit floating point
ti.init(arch=ti.gpu, fast_math=True)

@ti.kernel
def transpose_kernel(src: ti.types.ndarray(), dst: ti.types.ndarray()):
    for i, j in ti.ndrange(src.shape[0], src.shape[1]):
        dst[j, i] = src[i, j]


block_size: int = 32
@ti.kernel
def transpose_kernel(src: ti.types.ndarray(), dst: ti.types.ndarray()):
    # We initialize block with size (block_size, block_size)
    ti.loop_config(block_dim=block_size*block_size)
    for i, j in ti.ndrange(src.shape[0], src.shape[1]):
        # Step 1 Define shared memory tile (plus padding to avoid bank conflicts)
        tile = ti.simt.block.SharedArray(shape=(block_size, block_size+1), dtype=ti.f32)

        # Step 2 Calculate local indices within the block
        local_i = i % block_size
        local_j = j % block_size

        # Step 3 Load from global memory (src) to shared memory (tile)
        tile[local_i, local_j] = src[i, j]

        # Step 4 Sync threads to ensure the whole tile is loaded
        ti.simt.block.sync()

        # Step 5 Calculate transposed global indices
        block_i = i // block_size
        block_j = j // block_size
        new_i = block_j * block_size + local_i
        new_j = block_i * block_size + local_j

        # 6. Write back to global memory (dst) contiguously
        dst[new_i, new_j] = tile[local_j, local_i]


# Kernel annotation to force compile-time evaluation of the function.
@ti.kernel
def taichi_forward(x: ti.types.ndarray(), W_T: ti.types.ndarray(), b: ti.types.ndarray(), y: ti.types.ndarray()):
    for i, j in ti.ndrange(x.shape[0], W_T.shape[0]):
        # Define the size of the group of threads to be executed concurrently on the GPU.
        ti.loop_config(block_dim=64)
        acc = b[j]

        # Taichi lang vectorization of this loop is more efficient that list comprehension
        # Implementation of the linear layer
        for k in range(x.shape[1]):
            acc += x[i, k] * W_T[k, i]
        y[i, j] = acc

@ti.kernel
def taichi_backward(x: ti.types.ndarray(),   # Input values
                    W: ti.types.ndarray(),    # Weights matrix
                    dy: ti.types.ndarray(),   # Derivative of output value
                    dX: ti.types.ndarray(),   # Derivative input value
                    dW: ti.types.ndarray(),   # Derivative weight
                    db: ti.types.ndarray()):  # Derivative bias

    # Step 1: Propagate the diff dy  dX = SUM(dy * W)
    for i, k in ti.ndrange(dX.shape[0], dX.shape[1]):
        acc = 0.0
        ti.loop_config(block_dim=64)
        for j in range(dy.shape[1]):
            acc += dy[i, j] * W[j, k]
        dX[i, k] = acc

    # Step 2: Weights derivative dW = SUM(dy * x)
    for j, k in ti.ndrange(dW.shape[0], dW.shape[1]):
        acc = 0.0
        ti.loop_config(block_dim=64)
        for i in range(x.shape[0]):
            acc += dy[i, j] * x[i, k]
        dW[j, k] = acc

    # Step 3: Bias derivative db = SUM(dy)
    for j in range(db.shape[0]):
        acc = 0.0
        for i in range(dy.shape[0]):
            acc += dy[i, j]
        db[j] = acc


# Annotation required to link object attributes with the Taichi kernel
@ti.data_oriented
class TaichiDenseFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b):
        # Enforce contiguous memory allocation for input, weights and bias
        x_ = x.contiguous().float()
        W_ = W.contiguous().float()
        b_ = b.contiguous().float()

        # Initialize the output
        y = torch.empty((x_.shape[0], W_.shape[0]), device=x_.device, dtype=x_.dtype)

        # We allocated memory for the transpose of the weight matrix
        W_T = ti.ndarray(dtype=ti.f32, shape=(W_.shape[0], W_.shape[1]))
        # Apply the transposition
        transpose_kernel(W, W_T)
        # Invoke the kernel
        taichi_forward(x_, W_T, b_, y)
        # Similar to PyTorch - store the forward computation for the back propagation
        ctx.save_for_backward(x_, W_)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        # Load the parameters stored during the forward pass
        x_, W_ = ctx.saved_tensors
        dy = grad_out.contiguous().float()
        # Initialize the derivatives
        dX = torch.empty_like(x_)
        dW = torch.empty_like(W_)
        db = torch.empty((W_.shape[0],), device=W_.device, dtype=W_.dtype)
        # Apply back propagation
        taichi_backward(x_, W_, dy, dX, dW, db)
        return dX, dW, db


# Bridging PyTorch module hierarchy with Taichi Kernel
class TaichiLinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, scale_factor: float = 0.25) -> None:
        super().__init__()
        # Initialization of model parameters
        self.W = torch.nn.Parameter(torch.randn(output_dim, input_dim) * scale_factor)
        self.b = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate __call__ invocation to Taichi Autograd implementation
        return TaichiDenseFunction.apply(x, self.W, self.b)


class MLPTaichiModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_1_dim: int, hidden_2_dim: int, output_dim: int) -> None:
        super().__init__()
        # Replicate the PyTorch sequential modules
        self.fc1 = TaichiLinear(input_dim, hidden_1_dim)
        self.act1 = torch.nn.ReLU()
        self.fc2 = TaichiLinear(hidden_1_dim, hidden_2_dim)
        self.act2 = torch.nn.ReLU()
        self.fc3 = TaichiLinear(hidden_2_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))

"""
    Simple application using Pytorch
"""

class MLPTorchModel(nn.Module):
    def __init__(self, input_dim: int, hidden_1_dim: int, hidden_2_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1_dim),
            nn.ReLU(),
            nn.Linear(hidden_1_dim, hidden_2_dim),
            nn.ReLU(),
            nn.Linear(hidden_2_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForestCoverModelTraining(object):
    def __init__(self, model: nn.Module, num_epochs: int, learning_rate: float, display_loss: bool = False) -> None:
        self.model = model
        self.X_train, self.y_train, self.X_test, self.y_test = ForestCoverModelTraining.load_data()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.display_loss = display_loss

    @staticmethod
    def load_data() -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # Load dataset
        X, y = fetch_covtype(return_X_y=True)
        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y-1, test_size=0.15, random_state=42
        )

        # Standardize features (very important for MLPs)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        return X_train, y_train, X_test, y_test

    def train(self) -> List[float]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        durations = []
        start = time.time()
        for epoch in range(self.num_epochs):
            self.model.train()

            logits = self.model(self.X_train)
            loss = criterion(logits, self.y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            durations.append(time.time()-start)
            if self.display_loss and (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f}")
        return durations

    def eval(self) -> None:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.X_test).argmax(dim=1)
            accuracy = (prediction == self.y_test).float().mean()
        logging.info(f"Test accuracy: {accuracy:.3f}")


if __name__ == '__main__':
    import time

    forest_cover_torch_model = MLPTorchModel(input_dim=54, hidden_1_dim=196, hidden_2_dim=48, output_dim=7)
    forest_cover_taichi_model = MLPTaichiModel(input_dim=54, hidden_1_dim=196, hidden_2_dim=48, output_dim=7)

    forest_cover_torch_model_training = ForestCoverModelTraining(model=forest_cover_torch_model,
                                                                 num_epochs=12,
                                                                 learning_rate=1e-3,
                                                                 display_loss=True)
    # start = time.time()
    durations = forest_cover_torch_model_training.train()
    logging.info(durations)
    forest_cover_torch_model_training.eval()
    # duration = 'Torch wine model: {:.3f}'.format(time.time() - start)

    forest_cover_taichi_model_training = ForestCoverModelTraining(model=forest_cover_taichi_model,
                                                                  num_epochs=12,
                                                                  learning_rate=8e-3,
                                                                  display_loss=True)
    start = time.time()
    durations = forest_cover_taichi_model_training.train()
    logging.info(durations)
    forest_cover_taichi_model_training.eval()


