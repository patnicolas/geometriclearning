import unittest
import logging
import torch
from util.monitor_memory_device import monitor_memory_device
from util import torch_device
from python import memory_monitor_enabled
import python


class ProcessorMemoryTest(unittest.TestCase):
    # @unittest.skip('Ignored')
    def test_mps(self):

        @monitor_memory_device
        def heavy_test(limit: int):
            x = torch.Tensor([i for i in range(limit)])
            return x.to(torch.device('mps'))

        def outer() -> None:
            for i in range(1, 4):
                if memory_monitor_enabled:
                    _, report = heavy_test(i*5_000_000)
                    logging.info(report)
                else:
                    x = heavy_test(i * 5_000_000)

        logging.info(f'Device: {torch_device}')
        outer()

    @unittest.skip('Ignored')
    def test_mps_2(self):
        @monitor_memory_device
        def heavy_test():
            x = torch.Tensor([i for i in range(3_200_000)])
            logging.info(x)
            return True

        logging.info(f'Device: {torch_device}')
        _, report = heavy_test()
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

