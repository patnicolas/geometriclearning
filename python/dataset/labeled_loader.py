from peewee import NotSupportedError

_author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from typing import Callable, AnyStr, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset.base_loader import BaseLoader
from dataset.default_loader_generator import DefaultLoaderGenerator
import numpy as np

__all__ = ['LabeledLoader']

"""
    Wraps static methods to load public data sets. The methods generate two data loader
    - Training 
    - Evaluation
"""


class LabeledLoader(BaseLoader):
    type_dict = {'float64': np.float64, 'float32': np.float32, 'float16': np.float16, 'double': np.float64}

    def __init__(self,
                 create_dataset: Callable[[torch.Tensor, torch.Tensor], Dataset],
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
        super(LabeledLoader, self).__init__(batch_size, num_samples)
        self.create_dataset = create_dataset
        self.split_ratio = split_ratio

    def create_data_loaders(self,
                            features: torch.Tensor,
                            labels: torch.Tensor,
                            norm_factors: Tuple[float, float]) -> (DataLoader, DataLoader):
        """
            Generate Training and evaluation data loader from a given input_tensor
            @param features: Features input_tensor
            @type features: Torch tensor
            @param labels: labels input_tensor
            @type labels: Torch tensor
            @param norm_factors: Tuple of normalization factors
            @type norm_factors: Tuple (float, float)
            @return: Pair of Data loader for training data and validation data
            @rtype: Tuple of data loader
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(norm_factors[0],), std=(norm_factors[1],)),
        ])
        dataset = self.create_dataset(features, labels)
        return DefaultLoaderGenerator.generate_loader(dataset)

    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        raise NotImplementedError(f'Failed to load data from path {root_path}')
