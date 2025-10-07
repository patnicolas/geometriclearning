import unittest
import logging
import torch
from util.monitor_memory_device import monitor_memory_device
from util import torch_device
import python


class ProcessorMemoryTest(unittest.TestCase):
    # @unittest.skip('Ignored')
    def test_mps(self):
        @monitor_memory_device
        def heavy_test():
            x = torch.Tensor([i for i in range(3_200_000)])
            return x.to(torch.device('mps'))

        logging.info(f'Device: {torch_device}')
        result, report = heavy_test()
        logging.info(report)

    @unittest.skip('Ignored')
    def test_cpu(self):
        @monitor_memory_device
        def heavy_test():
            x = torch.Tensor([i for i in range(5_000_000)])
            return x

        logging.info(f'Device: {torch_device}')
        result, report = heavy_test()
        logging.info(report)

