import unittest
from torch import nn
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder
from dl.dlexception import DLException
from dl.block import ConvException
from typing import Tuple


class Conv2DBlockBuilderTest(unittest.TestCase):
    def test_call(self):
        output_channels = 15
        kernel_size = (4, 4)
        conv_2d_block_builder = Conv2DBlockBuilderTest.__create_conv_block(output_channels, kernel_size)
        modules = conv_2d_block_builder()
        for module in modules:
            print(str(module))

    def test_compute_out_channels(self):
        out_channels = 15
        kernel_size = (4, 4)
        try:
            conv_2d_block_builder = Conv2DBlockBuilderTest.__create_conv_block(out_channels, kernel_size)
            inferred_out_channels = conv_2d_block_builder.compute_out_shape()
            print(f'Inferred Out Channels {inferred_out_channels}')
            self.assertTrue(True)
        except ConvException as e:
            print(f'Failed: {str(e)}')
            self.assertTrue(False)

    def test_compute_out_channels_incorrect(self):
        out_channels = 15
        kernel_size = (6, 6)
        try:
            conv_2d_block_builder = Conv2DBlockBuilderTest.__create_conv_block(out_channels, kernel_size)
            inferred_out_channels = conv_2d_block_builder.compute_out_shape()
            print(f'Inferred Out Channels {inferred_out_channels[0]}')
            self.assertTrue(True)
        except ConvException as e:
            print(f'Failed: {str(e)}')
            self.assertTrue(False)

    def test_validate_succeed(self):
        out_channels = 15
        kernel_size = (4, 4)
        try:
            conv_2d_block_builder = Conv2DBlockBuilderTest.__create_conv_block(out_channels, kernel_size)
            assert (conv_2d_block_builder.is_valid())
            print(f'Inferred Out Channels {conv_2d_block_builder.is_valid()}')
        except ConvException as e:
            print(f'Failed: {str(e)}')
            self.assertTrue(False)

    def test_validate_failed(self):
        out_channels = 12
        kernel_size = (4, 4)
        try:
            conv_2d_block_builder = Conv2DBlockBuilderTest.__create_conv_block(out_channels, kernel_size)
            assert (not conv_2d_block_builder.is_valid())
            print(f'Inferred Out Channels {conv_2d_block_builder.is_valid()}')
        except ConvException as e:
            print(f'Failed: {str(e)}')
            self.assertTrue(False)

    @staticmethod
    def __create_conv_block(output_channels: int, kernel_size: Tuple[int, int]) -> Conv2DBlockBuilder:
        input_channels = 64
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        has_bias = False
        stride = (2, 2)
        padding = (2, 2)
        input_size = (28, 28)

        return Conv2DBlockBuilder(
            input_channels,
            output_channels,
            input_size,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            has_bias)


if __name__ == '__main__':
    unittest.main()
