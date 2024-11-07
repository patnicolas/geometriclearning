import unittest
from torch import nn
from dl.block.builder.deconv1d_block_builder import DeConv1DBlockBuilder
from dl.dl_exception import DLException


class DeConv1DBlockBuilderTest(unittest.TestCase):
    def test_init_succeed(self):
        kernel_size = 4
        output_channels = 16
        try:
            deconv_block_builder = DeConv1DBlockBuilderTest.__create_de_conv_block(kernel_size, output_channels)
            print(repr(deconv_block_builder))
            assert True
        except DLException as e:
            assert False

    def test_init_fails(self):
        kernel_size = 4
        output_channels = 16
        try:
            deconv_block_builder = DeConv1DBlockBuilderTest.__create_de_conv_block(kernel_size, output_channels)
            print(repr(deconv_block_builder))
            assert False
        except DLException as e:
            assert True

    @staticmethod
    def __create_de_conv_block(kernel_size: int, output_channels: int) -> DeConv1DBlockBuilder:
        input_channels = 62
        is_batch_normalization = True
        activation = nn.Tanh()
        has_bias = False
        stride = 4
        padding = 2

        return DeConv1DBlockBuilder(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            activation,
            has_bias)

