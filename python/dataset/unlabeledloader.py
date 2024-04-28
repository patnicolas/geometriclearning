__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import pandas as pd
from typing import Tuple, AnyStr, Callable, Optional
from python.dataset.unlabeleddataset import UnlabeledDataset
from python.dataset.tloader import TLoader

"""
    Wraps static methods to load public data sets. The methods generate two data loader
    - Training 
    - Evaluation
"""


class UnlabeledLoader(TLoader):
    def __init__(self, batch_size: int, split_ratio: float, num_samples: int = -1):
        """
        Constructor for this generic data set loader. A sub-sample is selected if num_samples is > 0 or
        the entire data set otherwise.

        @param batch_size: Size of the batch used in the loader
        @type batch_size: int
        @param num_samples: Number of samples loaded (or all data if num_samples <= 0)
        @type num_samples: int
        @param split_ratio: Training-validation random split ratio
        @type split_ratio: float
        """
        super(UnlabeledLoader, self).__init__(batch_size, split_ratio, num_samples)

    def load_mnist(self, norm_factors: list) -> (DataLoader, DataLoader):
        """
            Load MNIST library of digits
            :param norm_factors: List of two normalization factors
            :return: Pair Data loader for training data and validation data
        """
        assert len(norm_factors) == 2, f'Number of normalization factors {len(norm_factors)} should be 2'

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((norm_factors[0],), (norm_factors[1],)),
        ])
        mnist_dataset = MNIST('../data/', download=True, transform=transform)
        return self._generate_loader(mnist_dataset)

    def from_tensor(self, data: torch.Tensor, norm_factors: Tuple[float, float]) -> (DataLoader, DataLoader):
        """
            Generate Training and evaluation data loader from a given input_tensor
            @param data: Input input_tensor
            @type data: Torch tensor
            @param norm_factors: Tuple of normalization factors
            @type norm_factors: Tuple (float, float)
            @return: Pair of Data loader for training data and validation data
            @rtype: Tuple of data loader
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((norm_factors[0],), (norm_factors[1],)),
        ])
        dataset = UnlabeledDataset(data, transform)
        return self._generate_loader(dataset)

    def from_dataframe(self, df: pd.DataFrame, transform: Optional[Callable] = None) -> (DataLoader, DataLoader):
        dataset = UnlabeledDataset.from_df(df, transform)
        return self._generate_loader(dataset)

    def from_tensor_transform(self,
                              data: torch.Tensor,
                              transform: Optional[Callable] = None) -> (DataLoader, DataLoader):
        """
        Generate Training and evaluation data loader from a given input_tensor
        @param data: Input input_tensor
        @type data: Torch tensor
        @param transform: Optional pre-processing transform
        @type transform: Callable
        @return: Pair of Data loader for training data and validation data
        @rtype: Tuple of data loader
        """
        dataset = UnlabeledDataset(data, transform)
        return self._generate_loader(dataset)

