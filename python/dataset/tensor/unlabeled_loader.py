__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from torch.utils.data import DataLoader, Dataset
from dataset.default_loader_generator import DefaultLoaderGenerator
from torchvision import transforms
from typing import Tuple, AnyStr, Callable
from torchvision.transforms import Compose
from dataset.base_loader import BaseLoader

__all__ = ['UnlabeledLoader']

"""
    Wraps static methods to load public data sets. The methods generate two data loader
    - Training 
    - Evaluation
"""

class UnlabeledLoader(BaseLoader):
    def __init__(self,
                 create_dataset: Callable[[torch.Tensor, Compose], Dataset],
                 batch_size: int,
                 split_ratio: float,
                 num_samples: int = -1):
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
        super(UnlabeledLoader, self).__init__(batch_size, num_samples)
        self.split_ratio = split_ratio
        self.create_dataset = create_dataset

    def from_tensor(self,
                    data: torch.Tensor,
                    norm_factors: Tuple[float, float]) -> (DataLoader, DataLoader):
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
        dataset: Dataset = self.create_dataset(data, transform)
        return DefaultLoaderGenerator.generate_loader(dataset=dataset,
                                                      num_samples=self.num_samples,
                                                      batch_size=self.batch_size,
                                                      split_ratio=self.split_ratio)

    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        raise NotImplementedError(f'Failed to load data from path {root_path}')

