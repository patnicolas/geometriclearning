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

# Standard Library imports
from typing import AnyStr, Self
import logging
# 3rd Party imports
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import python
__all__ = ['ExecConfig']

class ExecConfig(object):
    def __init__(self,
                 empty_cache: bool,
                 mix_precision: bool,
                 subset_size: int,
                 monitor_memory: bool,
                 grad_accu_steps: int = 1,
                 pin_mem: bool = True):
        """
        Constructor for the configuration of the execution of training of DL models
        @param empty_cache: Flat to empty cache between epochs
        @type empty_cache: bool
        @param mix_precision: Flag to support upport mix precision Float16 for data and Float32 for model
        @type mix_precision: bool
        @param subset_size: Select a subset of the training data is >0, or entire data set if -1
        @type subset_size: int
        @param grad_accu_steps: Accumulate the computation of the gradient if > 0
        @type grad_accu_steps: in
        @param device_config: Device {'cpu', 'cuda' ..}
        @type device_config: str
        @param pin_mem: Flag to enable pin memory for data set loader
        @type pin_mem: bool
        """
        self.empty_cache = empty_cache
        self.mix_precision = mix_precision
        self.subset_size = subset_size
        self.grad_accu_steps = grad_accu_steps
        self.pin_mem = pin_mem
        self.device_config = None
        self.monitor_memory = monitor_memory
        self.accumulator = []

    @classmethod
    def default(cls) -> Self:
        """
        Default setting for the configuration of execution of training: All optimization are disabled
        @return: Instance of Execution configuration
        @rtype: ExecConfig
        """
        return cls(empty_cache=True,
                   mix_precision=False,
                   subset_size=0,
                   monitor_memory=True,
                   grad_accu_steps=1,
                   pin_mem=False)

    def __str__(self) -> AnyStr:
        return (f'\nEmpty cache: {self.empty_cache}\nMix precision {self.mix_precision}\nSubset size: {self.subset_size}'
                f'\ngrad_accu_steps: {self.grad_accu_steps}\nPin memory: {self.pin_mem}')

    def apply_monitor_memory(self) -> None:
        if self.monitor_memory and (self.device_config == 'mps' or self.device_config is None):
            allocated_mem = torch.mps.current_allocated_memory()
            total_mem = torch.mps.driver_allocated_memory()
            usage = 100.0*allocated_mem/total_mem

            self.accumulator.append(usage)
            logging.info(f'\nAllocated MPS: {format(allocated_mem, ",")}'
                         f'\nTotal MPS:     {format(total_mem, ",")}'
                         f'\nUsage MPS:     {usage:.2f}'
            )

    def apply_grad_accu_steps(self, idx: int, optimizer: Optimizer) -> None:
        if self.grad_accu_steps == 1 or (idx+1) % self.grad_accu_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def apply_mix_precision(self, x: torch.Tensor) -> torch.Tensor:
        return x.half() if self.mix_precision else x

    def apply_empty_cache(self) -> None:
        if self.empty_cache:
            torch.mps.empty_cache()

    def apply_batch_optimization(self, idx: int, optimizer: Optimizer) -> None:
        self.apply_empty_cache()
        self.apply_grad_accu_steps(idx, optimizer)

    def apply_optimize_loaders(self,
                               batch_size: int,
                               train_dataset: Dataset,
                               test_dataset: Dataset) -> (DataLoader, DataLoader):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            pin_memory=self.pin_mem,
            shuffle=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            pin_memory=self.pin_mem,
            shuffle=False)
        return train_loader, test_loader

    def apply_sampling(self,
                       train_dataset: Dataset,
                       test_dataset: Dataset) -> (Dataset, Dataset):
        if self.subset_size > 0:
            from torch.utils.data import Subset

            # Rescale the size of training and test data
            test_subset_size = int(float(self.subset_size * len(test_dataset)) / len(train_dataset))
            train_subset_size = self.subset_size - test_subset_size

            train_dataset = Subset(train_dataset, indices=range(train_subset_size))
            test_dataset = Subset(test_dataset, indices=range(test_subset_size))

        return train_dataset, test_dataset

    def apply_data_loaders(self,
                           batch_size: int,
                           train_dataset: Dataset,
                           eval_dataset: Dataset) -> (DataLoader, DataLoader):
        train_dataset, eval_dataset = self.apply_sampling(train_dataset, eval_dataset)
        return self.apply_optimize_loaders(batch_size, train_dataset, eval_dataset)

    def apply_labels_dtype(self, x: torch.Tensor, convert_to_float: bool = True) -> torch.Tensor:
        return (x.float() if self.device_config == 'mps' else x) if convert_to_float else x

    def apply_device(self) -> (AnyStr, torch.device):
        """
        Either pre-select or retrieve the device (CPU, GPU) used for the execution, training and evaluation
        of this Deep Learning model.
        @return: Pair (device name, torch device)
        @rtype: Tuple[AnyStr, torch.device]
        """
        assert self.device_config is None or self.device_config in ['auto', 'cpu', 'mps', 'cuda'], \
            f'Device {self.device_config} is not supported'

        if self.device_config is None:
            if torch.cuda.is_available():
                logging.info("Using CUDA GPU")
                return 'cuda', torch.device("cuda")
            elif torch.backends.mps.is_available():
                logging.info("Using MPS GPU")
                return 'mps', torch.device("mps")
            else:
                logging.info("Using CPU")
                return 'cpu', torch.device("cpu")
        else:
            logging.info(f'Using {self.device_config}')
            return self.device_config, torch.device(self.device_config)


