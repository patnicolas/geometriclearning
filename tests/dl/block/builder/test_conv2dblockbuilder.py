import unittest
from torch import nn
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder


class Conv2DBlockBuilderTest(unittest.TestCase):
    def test_call(self):
        conv_1d_block_builder = Conv2DBlockBuilderTest.__create_conv_block()
        modules = conv_1d_block_builder()
        for module in modules:
            print(str(module))

    @staticmethod
    def __create_conv_block() -> Conv2DBlockBuilder:
        input_channels = 64
        output_channels = 32
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        kernel_size = 3
        has_bias = False
        stride = 1
        padding = 1,

        return Conv2DBlockBuilder(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            has_bias)


if __name__ == '__main__':
    unittest.main()
