__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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
from typing import Callable, List, AnyStr
# 3rd Party libraries imports
import numpy as np
import torch
# Library imports
from deeplearning.loss.sgld import SGLD
from play import Play


"""
    Test loss landscape functions to evaluate optimization methods.
    These functions takes model parameters or weights w as input and return the loss function
"""

def ackley_func(w: torch.Tensor) -> torch.Tensor:
    # lr: 0.2, 1.8, 1.8
    return (-20.0 * torch.exp(-0.3 * torch.sqrt(torch.sum(w ** 2, dim=-1))) -
            torch.exp(torch.sum(torch.cos(w * 2 * torch.pi))) +
            torch.e +
            20.0)

def quadratic_func(w: torch.Tensor) -> torch.Tensor:
    # lr: 0.01, 0.03, 0.02
    return torch.sum(torch.pow(w, 2))


def rosenbrock_func(w: torch.Tensor) -> torch.Tensor:
    b = 100.0
    return (1 - w[0])**3 + b*(w[1] - w[0] ** 2)**2


class SGLDEvalPlay(Play):
    """
    Wrapper to implement the evaluation of Stochastic Gradient Langevin Dynamics as defined in Substack article:
    "Stochastic Gradient Langevin Dynamics"

    References:
    - Article:
    - Implementation: https://github.com/patnicolas/geometriclearning/blob/main/python/deeplearning/loss/sgld.py
    - Evaluation:  https://github.com/patnicolas/geometriclearning/blob/main/play/sgld_eval_play.py

    The features are implemented by the class SGLD in the source file
                       python/deeplearning/loss/sgld.py
    The class SGLDEvalPlay is a wrapper of the class SGLD
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self,
                 loss_function: Callable[[torch.Tensor], torch.Tensor],
                 loss_function_name: AnyStr) -> None:
        """
        Constructor for the evaluation of Stochastic Gradient Langevin Dynamics
        @param loss_function: Test loss function f(w) -> loss
        @type loss_function: Callable[[torch.Tensor], torch.Tensor]
        @param loss_function_name: Name of the test loss function for plotting purpose
        @type loss_function_name: str
        """
        super(SGLDEvalPlay, self).__init__()
        self.loss_function = loss_function
        self.loss_function_name = loss_function_name

    def play(self) -> None:
        self.__compare()
        self.__impact_learning_rate()

    """ ----------------------  Private Helper Methods --------------  """

    def __compare(self):
        w_sgd = torch.tensor([5.0, 5.0], requires_grad=True)
        w_adam = w_sgd.clone().detach().requires_grad_(True)
        w_sgld = w_sgd.clone().detach().requires_grad_(True)

        opt_sgd = torch.optim.SGD(params=[w_sgd], lr=0.2, momentum=0.9)
        opt_adam = torch.optim.Adam(params=[w_adam], lr=0.48)
        opt_sgld = SGLD(params=[w_sgld], lr=0.4)

        sgd_losses = []
        adam_losses = []
        sgld_losses = []
        for i in range(256):
            # SGD
            sgd_losses.append(self.__update(opt_sgd, w_sgd))
            # ADAM
            adam_losses.append(self.__update(opt_adam, w_adam))
            # SGLD
            sgld_losses.append(self.__update(opt_sgld, w_sgld))

        print(sgd_losses)
        print(adam_losses)
        print(sgld_losses)
        SGLDEvalPlay.plot(sgd_losses, adam_losses, sgld_losses, self.loss_function_name)

    def __impact_learning_rate(self):
        w_sgld1 = torch.tensor([5.0, 5.0], requires_grad=True)
        w_sgld2 = w_sgld1.clone().detach().requires_grad_(True)
        w_sgld3 = w_sgld1.clone().detach().requires_grad_(True)

        opt_sgld1 = SGLD(params=[w_sgld1], lr=0.05)
        opt_sgld2 = SGLD(params=[w_sgld2], lr=0.2)
        opt_sgld3 = SGLD(params=[w_sgld3], lr=0.8)

        sgd_losses = []
        adam_losses = []
        sgld_losses = []
        for i in range(512):
            # SGD
            sgd_losses.append(self.__update(opt_sgld1, w_sgld1))
            # ADAM
            adam_losses.append(self.__update(opt_sgld2, w_sgld2))
            # SGLD
            sgld_losses.append(self.__update(opt_sgld3, w_sgld3))

        print(sgd_losses)
        print(adam_losses)
        print(sgld_losses)
        SGLDEvalPlay.plot(sgd_losses, adam_losses, sgld_losses, self.loss_function_name)

    def __update(self, optimizer: torch.optim.Optimizer, w_data: torch.Tensor) -> np.ndarray:
        """
        Implement the update step for optimizer - standard implementation

        @param optimizer: optimizer
        @type optimizer: Optimizer
        @param w_data: weight data
        @type w_data: torch.Tensor
        @return: numpy array as pair (model parameters, loss value)
        @rtype: np.ndarray
        """
        optimizer.zero_grad()
        loss = self.loss_function(w_data)
        loss.backward()
        optimizer.step()
        return np.append(w_data.detach().clone().numpy(), loss.detach().clone().numpy())

    @staticmethod
    def plot(sgd_losses: List[np.ndarray],
             adam_losses: List[np.ndarray],
             sgld_losses: List[np.ndarray],
             sampling_function: AnyStr) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = zip(*sgd_losses)
        ax.plot(x, y, z, color='blue', linewidth=2, alpha=0.4, label="SGLD lr=0.05")
        x, y, z = zip(*adam_losses)
        ax.plot(x, y, z, color='red', linewidth=2, alpha=0.3, label="SGLD lr=0.2")
        x, y, z = zip(*sgld_losses)
        ax.plot(x, y, z, color='green', linewidth=2, alpha=0.7, label="SGLD lr=0.8")
        ax.set_xlabel('W1')
        ax.set_ylabel('W2')
        ax.set_zlabel('Loss')
        ax.set_title(f'3D Loss Trajectory Comparison - {sampling_function}')
        ax.legend()

        plt.show()


if __name__ == '__main__':
    sgld_eval = SGLDEvalPlay(quadratic_func, 'Quadratic')
    sgld_eval.play()
    sgld_eval = SGLDEvalPlay(rosenbrock_func, 'Rosenbrock')
    sgld_eval.play()
    sgld_eval = SGLDEvalPlay(ackley_func, 'Ackley')
    sgld_eval.play()
    sgld_eval = SGLDEvalPlay(rosenbrock_func, 'Rosenbrock')
    sgld_eval.play()




