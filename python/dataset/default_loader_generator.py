_author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


class DefaultLoaderGenerator(object):
    @staticmethod
    def generate_loader(dataset: Dataset,
                        num_samples: int,
                        split_ratio: float,
                        batch_size: int) -> (DataLoader, DataLoader):
        """
        Generate a basic loader for both labeled and unlabeled data loaders
        @param dataset: Dataset for which training and validation data loaders have to be created
        @type dataset: utils.data.Dataset
        @param num_samples: Number of samples
        @type num_samples: in
        @param split_ratio: Training/Validation sample split ratio
        @type split_ratio: float
        @param batch_size: Size of the batch of data points for trainiing
        @type batch_size: int
        @return: Pair training data loader, validation data loader
        @rtype: Tuple[DataLoader, DataLoader]
        """
        _dataset = torch.utils.data.Subset(dataset, np.arange(num_samples)) if num_samples > 0 \
            else dataset
        training_size = int(len(_dataset) * split_ratio)
        validation_size = len(_dataset) - training_size

        train_dataset, valid_dataset = random_split(dataset, lengths=(training_size, validation_size))
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        eval_data_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        return train_data_loader, eval_data_loader
