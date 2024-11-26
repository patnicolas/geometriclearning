__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from typing import AnyStr
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class ExecConfig(object):
    def __init__(self,
                 empty_cache: bool,
                 mix_precision: bool,
                 subset_size: int,
                 monitor_memory: bool,
                 grad_accu_steps: int = 1,
                 device_config: AnyStr = None,
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
        self.device_config = device_config
        self.pin_mem = pin_mem
        self.monitor_memory = monitor_memory
        self.accumulator = []

    def __str__(self) -> AnyStr:
        return (f'\nEmpty cache: {self.empty_cache}\nMix precision {self.mix_precision}\nSubset size: {self.subset_size}'
                f'\ngrad_accu_steps: {self.grad_accu_steps}\nDevice: {self.device_config}\nPin memory: {self.pin_mem}')

    def apply_monitor_memory(self) -> None:
        if self.monitor_memory and (self.device_config == 'mps' or self.device_config is None):
            allocated_mem = torch.mps.current_allocated_memory()
            total_mem = torch.mps.driver_allocated_memory()
            usage = 100.0*allocated_mem/total_mem

            self.accumulator.append(usage)
            print(f'\nAllocated MPS: {format(allocated_mem, ",")}'
                  f'\nTotal MPS:     {format(total_mem, ",")}'
                  f'\nUsage MPS:     {usage:.2f}'
            )

    def apply_grad_accu_steps(self, idx: int, optimizer: Optimizer) -> None:
        if self.grad_accu_steps == 1 or (idx+1) % self.grad_accu_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    def apply_mix_precision(self, x: torch.Tensor) -> torch.Tensor:
        return x.half() if self.mix_precision else x

    def apply_empty_cache(self) -> None:
        if self.empty_cache:
            torch.mps.empty_cache()

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
                print("Using CUDA GPU")
                return 'cuda', torch.device("cuda")
            elif torch.backends.mps.is_available():
                print("Using MPS GPU")
                return 'mps', torch.device("mps")
            else:
                print("Using CPU")
                return 'cpu', torch.device("cpu")
        else:
            print(f'Using {self.device_config}')
            return self.device_config, torch.device(self.device_config)


