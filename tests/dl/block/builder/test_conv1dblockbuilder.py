import unittest
from torch import nn
from dl.block.builder.conv1dblockbuilder import Conv1DBlockBuilder
from dl.dlexception import DLException


class Conv1DBlockBuilderTest(unittest.TestCase):
    def test_call(self):
        out_channels = 15
        kernel_size = 10
        conv_1d_block_builder = Conv1DBlockBuilderTest.__create_conv_block(kernel_size, out_channels)
        modules = conv_1d_block_builder()
        for module in modules:
            print(str(module))

    def test_compute_out_channels(self):
        out_channels = 15
        kernel_size = 10
        try:
            conv_1d_block_builder = Conv1DBlockBuilderTest.__create_conv_block(kernel_size, out_channels)
            inferred_out_channels = conv_1d_block_builder.compute_out_channels()
            assert(inferred_out_channels >= 0)
            print(f'Inferred Out Channels {inferred_out_channels}')
        except DLException as e:
            assert(False)


    def test_compute_out_channels_incorrect(self):
        out_channels = 15
        kernel_size = 8
        try:
            conv_1d_block_builder = Conv1DBlockBuilderTest.__create_conv_block(kernel_size, out_channels)
            inferred_out_channels = conv_1d_block_builder.compute_out_channels()
            assert(inferred_out_channels == -1)
            print(f'Inferred Out Channels {inferred_out_channels}')
        except DLException as e:
            assert (True)

    def test_validate_succeed(self):
        out_channels = 15
        kernel_size = 10
        try:
            conv_1d_block_builder = Conv1DBlockBuilderTest.__create_conv_block(kernel_size, out_channels)
            print(f'Inferred Out Channels {conv_1d_block_builder.is_valid()}')
        except DLException as e:
            assert (False)

    def test_validate_failed(self):
        out_channels = 12
        kernel_size = 10
        try:
            conv_1d_block_builder = Conv1DBlockBuilderTest.__create_conv_block(kernel_size, out_channels)
            print(f'Inferred Out Channels {conv_1d_block_builder.is_valid()}')
        except DLException as e:
            assert (True)

    @staticmethod
    def __create_conv_block(kernel_size: int, output_channels: int) -> Conv1DBlockBuilder:
        input_channels = 62
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        has_bias = False
        stride = 4
        padding = 2
        input_size = 28

        return Conv1DBlockBuilder(
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