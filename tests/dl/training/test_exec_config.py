import unittest
from typing import AnyStr
from dl.training.exec_config import ExecConfig
import logging

class ExecConfigTest(unittest.TestCase):

    def test_init(self):
        empty_cache: bool = True
        mix_precision: bool = False
        pin_memory: bool = True
        subset_size: int = -1
        device_config: AnyStr = 'cpu'
        monitor_memory = True
        grad_accu_steps: int = 1

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size,
            monitor_memory=monitor_memory,
            grad_accu_steps=grad_accu_steps,
            device_config=device_config)
        logging.info(exec_config)
        self.assertTrue(True)

    def test_get_device(self):
        empty_cache: bool = True
        mix_precision: bool = False
        pin_memory: bool = True
        subset_size: int = -1
        device_config: AnyStr = 'mps'
        monitor_memory = True
        grad_accu_steps: int = 1

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size,
            monitor_memory=monitor_memory,
            grad_accu_steps=grad_accu_steps,
            device_config=device_config)

        device_name, _ = exec_config.apply_device()
        exec_config.apply_monitor_memory()
        self.assertTrue(device_name == 'mps')

    def test_get_device_2(self):
        empty_cache: bool = True
        mix_precision: bool = False
        pin_memory: bool = True
        subset_size: int = -1
        monitor_memory = True
        grad_accu_steps: int = 1

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            monitor_memory=monitor_memory,
            grad_accu_steps=grad_accu_steps,
            subset_size=subset_size)

        device_name, _ = exec_config.apply_device()
        logging.info(f'Native device: {device_name}')
        self.assertTrue(True)

    def test_get_mix_precision(self):
        import torch
        empty_cache: bool = True
        mix_precision: bool = True
        pin_memory: bool = True
        subset_size: int = -1
        monitor_memory = True
        grad_accu_steps: int = 1

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            monitor_memory=monitor_memory,
            grad_accu_steps=grad_accu_steps,
            subset_size=subset_size)

        x = torch.Tensor([90.1, 98.2])
        logging.info(f'\nOriginal: {x.dtype}')
        self.assertTrue(x.dtype == torch.float32)
        x = exec_config.apply_mix_precision(x)
        self.assertTrue(x.dtype == torch.float16)
        logging.info(f'\nConverted: {x.dtype}')