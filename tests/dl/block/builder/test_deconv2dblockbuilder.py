import unittest
from torch import nn
from dl.block.builder.deconv2dblockbuilder import DeConv2DBlockBuilder
from dl.dlexception import DLException
from typing import Tuple

class DeConv2DBlockBuilderTest(unittest.TestCase):
    def test_init_succeed(self):
        kernel_size = (2, 2)
        output_channels = 16
        try:
            deconv_block_builder = DeConv2DBlockBuilderTest.__create_de_conv_block(kernel_size, output_channels)
            print(repr(deconv_block_builder))
            assert True
        except DLException as e:
            assert False

    def test_init_fails(self):
        kernel_size = (4, 4)
        output_channels = 16
        try:
            deconv_block_builder = DeConv2DBlockBuilderTest.__create_de_conv_block(kernel_size, output_channels)
            print(repr(deconv_block_builder))
            assert False
        except DLException as e:
            assert True

    @staticmethod
    def __create_de_conv_block(kernel_size: Tuple[int, int], output_channels: int) -> DeConv2DBlockBuilder:
        input_channels = 62
        is_batch_normalization = True
        activation = nn.Tanh()
        has_bias = False
        stride = (2, 2)
        padding = (2, 2)

        return DeConv2DBlockBuilder(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            activation,
            has_bias)

