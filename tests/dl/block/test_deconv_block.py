import unittest
from torch import nn
from dl.block.deconv_block import DeConvBlock
from dl.block.conv_block import ConvBlock
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder
from dl import ConvException
from typing import Tuple


class DeConvBlockTest(unittest.TestCase):
    @unittest.skip('Ignore')
    def test_init_de_conv1_succeeded(self):
        dimension = 1
        out_channels = 32
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            print(repr(conv_block))
            self.assertTrue(conv_block.out_channels == out_channels)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_from_conv_block(self):
        in_channels = 32
        conv_block = DeConvBlockTest.__create_conv_block(in_channels)
        de_conv_block = conv_block.invert_with_activation(nn.Sigmoid())
        print(f'\nConv block:\n{repr(conv_block)}\nDe conv block:\n{repr(de_conv_block)}')

    @unittest.skip('Ignore')
    def test_init_de_conv1_failed(self):
        dimension = 1
        out_channels = 24
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            print(repr(conv_block))
            self.assertTrue(conv_block.out_channels == 24)
        except ConvException as e:
            print(str(e))
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_init_de_conv2_succeed(self):
        dimension = 2
        out_channels = 71
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            self.assertTrue(conv_block.out_channels == 71)
            print(repr(conv_block))
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_de_conv2_failed(self):
        dimension = 2
        out_channels = 64
        try:
            conv_block = DeConvBlockTest.__create_de_conv_block(dimension, out_channels)
            self.assertFalse(conv_block.out_channels == 71)
            print(repr(conv_block))
        except ConvException as e:
            print(str(e))
            self.assertTrue(True)

    @staticmethod
    def __create_conv_block(out_channels: int) -> ConvBlock:
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()

        in_channels = 68
        kernel_size = (2, 2)
        stride = (2, 2)
        padding = (2, 2)
        input_size = (28, 28)
        conv_block_builder = Conv2DBlockBuilder(
                in_channels,
                out_channels,
                input_size,
                kernel_size,
                stride,
                padding,
                is_batch_normalization,
                max_pooling_kernel,
                activation,
                True)
        return ConvBlock('test', conv_block_builder)

    @staticmethod
    def __create_de_conv_block(dimension: int, out_channels: int | Tuple[int, int]) -> DeConvBlock:
        in_channels = 17
        is_batch_normalization = True
        activation = nn.ReLU()
        if dimension == 1:
            input_size = 32
            kernel_size = 4
            stride = 2
            padding = 2
        elif dimension == 2:
            input_size = (32, 32)
            kernel_size = (3, 3)
            stride = (2, 2)
            padding = (1, 1)
        else:
            raise ConvException(f'Dimension {dimension} is not supported')

        has_bias = False
        return DeConvBlock.build(
                dimension,
                in_channels,
                out_channels,
                input_size,
                kernel_size,
                stride,
                padding,
                is_batch_normalization,
                activation,
                bias=has_bias)


if __name__ == '__main__':
    unittest.main()