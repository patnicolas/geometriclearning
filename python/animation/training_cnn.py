__author__ = "Patrick Nicolas"
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

from manim import *
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2704, 10)  # Flattened after conv

    def forward(self, x):
        x = self.conv1(x)
        logging.info(x.shape)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        logging.info(x.shape)
        x = self.fc1(x)
        return x

# Training loop integrated with Manim scene
class TrainingCNN(Scene):
    def construct(self):
        # Title
        title = Text("CNN Training with PyTorch", font_size=36).to_edge(UP)
        self.add(title)
        #  self.play(Write(title))

        # Initialize model and optimizer
        model = SimpleCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Prepare dataset (MNIST subset)
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root="./", train=True, download=True, transform=transform)
        loader = DataLoader(train_data, batch_size=64, shuffle=True)

        # ValueTracker for loss
        loss_tracker = ValueTracker(2.0)
        loss_display = always_redraw(
            lambda: Tex(rf"\text{{Loss: }} {loss_tracker.get_value():.3f}").to_edge(DOWN)
        )
        self.add(loss_display)

        # Train for a few steps and visualize
        for i, (images, labels) in enumerate(loader):
            if i > 30:
                break
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            self.play(loss_tracker.animate.set_value(loss.item()), run_time=0.1)

        self.wait(2)

if __name__ == '__main__':
    scene = SimpleCNN()
    scene.construct()
