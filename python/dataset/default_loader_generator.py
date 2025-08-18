_author__ = "Patrick Nicolas"
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

# 3rd Party imports
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
__all__ = ['DefaultLoaderGenerator']


class DefaultLoaderGenerator(object):
    """
    Class that encapsulates a generic extractor of training and validation data loader given
    a dataset, num of samples, a split ration and a batch size
    """
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
