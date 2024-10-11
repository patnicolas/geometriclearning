import unittest
from torch import nn
from dl.block.deconvblock import DeConvBlock
from dl.dlexception import DLException
from typing import Tuple

class DeConvBlockTest(unittest.TestCase):
    def test_init_de_conv1_succeeded(self):
        dimension = 1
        out_channels = 32
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            print(repr(conv_block))
            self.assertTrue(conv_block.out_channels == out_channels)
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_de_conv1_failed(self):
        dimension = 1
        out_channels = 24
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            print(repr(conv_block))
            self.assertFalse(conv_block.out_channel == out_channels)
        except DLException as e:
            print(str(e))
            self.assertTrue(True)

    def test_init_de_conv2_succeed(self):
        dimension = 2
        out_channels = 71
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            self.assertTrue(conv_block.out_channels == 71)
            print(repr(conv_block))
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_de_conv2_failed(self):
        dimension = 2
        out_channels = 64
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            self.assertFalse(conv_block.out_channels == 71)
            print(repr(conv_block))
        except DLException as e:
            print(str(e))
            self.assertTrue(True)

    @staticmethod
    def __create_de_conv_block(dimension: int, out_channels: int | Tuple[int, int]) -> DeConvBlock:
        in_channels = 17
        is_batch_normalization = True
        activation = nn.ReLU()
        if dimension == 1:
            kernel_size = 4
            stride = 2
            padding = 2
        elif dimension == 2:
            kernel_size = (3, 3)
            stride = (2, 2)
            padding = (1, 1)
        else:
            raise DLException(f'Dimension {dimension} is not supported')

        has_bias = False
        return DeConvBlock(
                dimension,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                is_batch_normalization,
                activation,
                bias=has_bias)


if __name__ == '__main__':
    unittest.main()