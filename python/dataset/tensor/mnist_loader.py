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

from dataset.base_loader import BaseLoader
from typing import AnyStr
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from dl.model import GrayscaleToRGB
from dataset import DatasetException
import logging
import python


class MNISTLoader(BaseLoader):
    plot_layout = (2, 4)
    default_normalization = (0.5, 0.5, 0.5)

    def __init__(self, batch_size: int, resize_image: int = -1) -> None:
        """
        Constructor for the MNIST torch data loader
        @param batch_size: size of batch for loading the Caltech101 data set
        @param batch_size: 1
        @param resize_image: Image to be resize if positive value, otherwise the original image is preserved
        @type resize_image: int
        """
        assert 0 < batch_size <= 8192, f'Batch size {batch_size} should be [1, 8192]'
        assert 0 < resize_image <= 8192, f'Resize image factor {resize_image} should be [1, 8192]'

        super(MNISTLoader, self).__init__(batch_size=batch_size, num_samples=-1)
        self.resize_image = resize_image

    @staticmethod
    def show_samples(images_set: Dataset, is_random: bool = True) -> None:
        """
        Show a random sample of images from a given dataset
        @param images_set: Data set extracted from data loader
        @type images_set: Dataset
        @param is_random: Flag to specify the images have to be randomly selected
        @type is_random: bool
        """
        import random
        import matplotlib.pyplot as plt

        images = images_set.data
        labels = images_set.targets
        num_plots = MNISTLoader.plot_layout[0]*MNISTLoader.plot_layout[1]
        indices = [random.randint(a=0, b=len(labels)) if is_random else 0 for _ in range(num_plots)]
        fig, axes = plt.subplots(nrows=MNISTLoader.plot_layout[0], ncols=MNISTLoader.plot_layout[1], figsize=(12, 6))

        for idx, ax in enumerate(axes.flat):
            img = images[indices[idx]]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            title = str(int(labels[indices[idx]]))
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def extract_features(self, root_path: AnyStr) -> (Dataset, Dataset):
        train_dataset, eval_dataset = self._extract_datasets(root_path)
        return train_dataset, eval_dataset

    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to CIFAR10 data
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        try:
            transform = transforms.Compose([
                transforms.Resize(size =(self.resize_image, self.resize_image), interpolation=InterpolationMode.BILINEAR),
                GrayscaleToRGB(),
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                # Normalize with mean and std for RGB channels
                transforms.Normalize(mean=MNISTLoader.default_normalization, std=MNISTLoader.default_normalization)
            ]) if self.resize_image > 0 else transforms.Compose([
                GrayscaleToRGB(),
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                # Normalize with mean and std for RGB channels
                transforms.Normalize(mean=MNISTLoader.default_normalization, std=MNISTLoader.default_normalization)
            ])

            train_dataset = MNIST(
                root=root_path,  # Directory to store the dataset
                train=True,  # Load training data
                download=True,  # Download if not already present
                transform=transform  # Apply transformations
            )

            eval_dataset = MNIST(
                root=root_path,  # Directory to store the dataset
                train=False,  # Load test data
                download=True,  # Download if not already present
                transform=transform  # Apply transformations
            )
            return train_dataset, eval_dataset

        except (RuntimeError | ValueError | TypeError) as e:
            logging.error(str(e))
            raise DatasetException(str(e))