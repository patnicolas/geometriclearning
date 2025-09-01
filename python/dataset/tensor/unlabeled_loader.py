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

# Standard Library imports
from typing import Tuple, AnyStr, Callable
import logging
# 3rd Party imports
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose
# Library imports
from dataset.base_loader import BaseLoader
from dataset.default_loader_generator import DefaultLoaderGenerator
import python
from dataset import DatasetException

__all__ = ['UnlabeledLoader']

class UnlabeledLoader(BaseLoader):
    """
        Wraps static methods to load public data sets. The methods generate two data loader
        - Training
        - Evaluation
    """

    def __init__(self,
                 create_dataset: Callable[[torch.Tensor, Compose], Dataset],
                 batch_size: int,
                 split_ratio: float,
                 num_samples: int = -1) -> None:
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
        assert 0 < batch_size <= 8192, f'Batch size {batch_size} should be [1, 8192]'
        assert 0.5 <= split_ratio <= 0.95, f'Training-validation split ratio {split_ratio} should be [0.5, 0.95]'
        assert -2 < num_samples <= 1e+6 and num_samples != 0, f'Number of samples {num_samples} should be [1, 1e+6]'

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
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((norm_factors[0],), (norm_factors[1],)),
            ])
            dataset: Dataset = self.create_dataset(data, transform)
            return DefaultLoaderGenerator.generate_loader(dataset=dataset,
                                                          num_samples=self.num_samples,
                                                          batch_size=self.batch_size,
                                                          split_ratio=self.split_ratio)
        except (RuntimeError | ValueError | TypeError) as e:
            logging.error(str(e))
            raise DatasetException(str(e))

    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        raise NotImplementedError(f'Failed to load data from path {root_path}')

