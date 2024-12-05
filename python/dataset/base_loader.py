__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch.utils.data import DataLoader, Dataset
from dl.training.exec_config import ExecConfig
from typing import AnyStr
from dataset import DatasetException
import abc
from abc import ABC


class BaseLoader(ABC):
    def __init__(self, batch_size: int, num_samples: int = -1):
        """
        Constructor for this generic data set loader. A sub-sample is selected if num_samples is > 0 or
        the entire data set otherwise.

        @param batch_size: Size of the batch used in the loader
        @type batch_size: int
        @param num_samples: Number of samples loaded (or all data if num_samples <= 0)
        @type num_samples: int
        """
        assert batch_size >= 2, f'Batch size {batch_size} should be >= 4'
        self.batch_size = batch_size
        self.num_samples = num_samples

    def loaders_from_path(self, root_path: AnyStr, exec_config: ExecConfig) -> (DataLoader, DataLoader):
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
                              exec_config: ExecConfig) -> (DataLoader, DataLoader):
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
        train_loader, test_loader = exec_config.apply_data_loaders(self.batch_size, train_dataset, eval_dataset)
        return train_loader, test_loader

    def __str__(self) -> AnyStr:
        return f'Batch size: {self.batch_size}, Num samples: {self.num_samples}'

    @abc.abstractmethod
    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        raise DatasetException(f'Failed to load data from path {root_path}')

    """ --------------------  Helper method --------------------------- """
    """
    def _generate_loader(self, dataset: Dataset):
        _dataset = torch.utils.data.Subset(dataset,
                                           np.arange(self.num_samples),
                                           TDataset.numpy_dtype('float32')) if self.num_samples > 0 else dataset

        training_size = int(len(_dataset) * self.split_ratio)
        validation_size = len(_dataset) - training_size
        train_dataset, valid_dataset = random_split(dataset, (training_size, validation_size))
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        return train_data_loader, eval_data_loader
    """
