__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dl.model.custom.conv_2D_config import Conv2DConfig
from typing import AnyStr
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger('dl.model.custom.ConvCifar10')


class ConvCifar10(object):
    id = 'Convolutional_CIFAR10'

    def __init__(self, conv_2D_config: Conv2DConfig) -> None:
        self.model = conv_2D_config.conv_model

    def load_dataset(self, root_path: AnyStr) -> (DataLoader, DataLoader):
        train_dataset, test_dataset = self.__extract_datasets(root_path)

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader

    """ ---------------------------  Private helper methods ---------------------------- """
    def __extract_datasets(self, root_path: AnyStr) -> (CIFAR10, CIFAR10):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to CIFAR10 data
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        from dl.training.neural_net import NeuralNet

        _, torch_device = NeuralNet.get_device()

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))  # Normalize with mean and std for RGB channels
        ])

        train_dataset = CIFAR10(
            root=root_path,  # Directory to store the dataset
            train=True,  # Load training data
            download=True,  # Download if not already present
            transform=transform  # Apply transformations
        )

        test_dataset = CIFAR10(
            root=root_path,  # Directory to store the dataset
            train=False,  # Load test data
            download=True,  # Download if not already present
            transform=transform  # Apply transformations
        )

        return train_dataset, test_dataset





