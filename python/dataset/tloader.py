import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import AnyStr


class TLoader(object):
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
        assert batch_size >= 2, f'Batch size {batch_size} should be >= 4'
        assert 0.5 <= split_ratio <= 0.95, f'Training-validation split ratio {split_ratio} should be [0.5, 0.95]'

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.split_ratio = split_ratio

    def from_dataset(self, dataset: Dataset) -> (DataLoader, DataLoader):
        """
            Generate Training and evaluation data loader from a given dataset which type is inherited from Dataset
            @param dataset: input data set
            @type dataset: Sub-class of torch.utils.data.Dataset
            @return: Pair of Data loader for training data and validation data
            @rtype: Tuple of data loader
        """
        return self._generate_loader(dataset)

    def __repr__(self) -> AnyStr:
        return f'Batch size: {self.batch_size}, Num samples: {self.num_samples}, Train-eval split: {self.split_ratio}'


    def _generate_loader(self, dataset: Dataset):
        _dataset = torch.utils.data.Subset(dataset, np.arange(self.num_samples)) if self.num_samples > 0 \
            else dataset

        training_size = int(len(_dataset) * self.split_ratio)
        validation_size = len(_dataset) - training_size
        train_dataset, valid_dataset = random_split(dataset, (training_size, validation_size))
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        return train_data_loader, eval_data_loader
