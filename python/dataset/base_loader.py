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

from torch.utils.data import DataLoader, Dataset
from deeplearning.training.exec_config import ExecConfig
from typing import AnyStr
from dataset import DatasetException
import abc
from abc import ABC
__all__ = ['BaseLoader']


class BaseLoader(ABC):
    """
    Base class for various PyTorch and PyTorch Geometric loader
    """
    def __init__(self, batch_size: int, num_samples: int = -1):
        """
        Constructor for this generic data set loader. A sub-sample is selected if num_samples is > 0 or
        the entire data set otherwise.

        @param batch_size: Size of the batch used in the loader
        @type batch_size: int
        @param num_samples: Number of samples loaded (or all data if num_samples <= 0)
        @type num_samples: int
        """
        assert 2 <= batch_size <= 8192, f'Batch size {batch_size} should be [2, 8192]'

        self.batch_size = batch_size
        self.num_samples = num_samples

    def loaders_from_path(self,
                          root_path: AnyStr,
                          exec_config: ExecConfig = ExecConfig.default()) -> (DataLoader, DataLoader):
        """
        Create Torch loaders for training and evaluation data set from either the local file or default HTTP server
        @param root_path: Relative path for the local data set
        @type root_path: str
        @param exec_config: Configuration to optimize the loading of data
        @type exec_config: ExecConfig
        @return: Pair of training data loader, evaluation data loader
        @rtype: Tuple[DataLoader, DataLoader]
        """
        train_dataset, eval_dataset = self._extract_datasets(root_path)
        return self.loaders_from_datasets(train_dataset, eval_dataset, exec_config)

    def loaders_from_datasets(self,
                              train_dataset: Dataset,
                              eval_dataset: Dataset,
                              exec_config: ExecConfig = ExecConfig.default()) -> (DataLoader, DataLoader):
        """
        Create Torch loaders for training and evaluation data set from the training and evaluation data sets
        @param train_dataset: Training torch data set
        @type train_dataset: Dataset
        @param eval_dataset: Evaluation torch data set
        @type eval_dataset: Dataset
        @param exec_config: Configuration to optimize the loading of data
        @type exec_config: ExecConfig
        @return: Pair of training data loader, evaluation data loader
        @rtype: Tuple[DataLoader, DataLoader]
        """
        # Create DataLoaders for batch processing
        train_loader, eval_loader = exec_config.apply_data_loaders(self.batch_size, train_dataset, eval_dataset)
        return train_loader, eval_loader

    def __str__(self) -> AnyStr:
        return f'Batch size: {self.batch_size}, Num samples: {self.num_samples}'

    @abc.abstractmethod
    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        """
        Abstract protected method for extracting a training and validation dataset from a file
        @param root_path: Absolute or relative path for the file containing the dataset
        @type root_path:
        @return: Pair of training and validation data sets
        @rtype: Tuple[Dataset, Dataset]
        """
        raise DatasetException(f'Failed to load data from path {root_path}')
