_author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


class DefaultLoaderGenerator(object):
    @staticmethod
    def generate_loader(self, dataset: Dataset) -> (DataLoader, DataLoader):
        _dataset = torch.utils.data.Subset(dataset, np.arange(self.num_samples)) if self.num_samples > 0 \
            else dataset
        training_size = int(len(_dataset) * self.split_ratio)
        validation_size = len(_dataset) - training_size

        train_dataset, valid_dataset = random_split(dataset, lengths=(training_size, validation_size))
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        return train_data_loader, eval_data_loader
