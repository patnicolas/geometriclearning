import unittest
from typing import AnyStr
from dl.training.exec_config import ExecConfig

class ExecConfigTest(unittest.TestCase):

    def test_init(self):
        empty_cache: bool = True
        mix_precision: bool = False
        pin_memory: bool = True
        subset_size: int = -1
        device_config: AnyStr = 'cpu'

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size,
            device_config=device_config)
        print(exec_config)
        self.assertTrue(True)

    def test_get_device(self):
        empty_cache: bool = True
        mix_precision: bool = False
        pin_memory: bool = True
        subset_size: int = -1
        device_config: AnyStr = 'cuda'

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size,
            device_config=device_config)
        device_name, _ = exec_config.get_device()
        self.assertTrue(device_name == 'cuda')

    def test_get_device_2(self):
        empty_cache: bool = True
        mix_precision: bool = False
        pin_memory: bool = True
        subset_size: int = -1

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size)
        device_name, _ = exec_config.get_device()
        print(f'Native device: {device_name}')
        self.assertTrue(True)

    def test_get_mix_precision(self):
        import torch
        empty_cache: bool = True
        mix_precision: bool = True
        pin_memory: bool = True
        subset_size: int = -1

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size)
        x = torch.Tensor([90.1, 98.2])
        print(f'\nOriginal: {x.dtype}')
        self.assertTrue(x.dtype == torch.float32)
        x = exec_config.apply_mix_precision(x)
        self.assertTrue(x.dtype == torch.float16)
        print(f'\nConverted: {x.dtype}')