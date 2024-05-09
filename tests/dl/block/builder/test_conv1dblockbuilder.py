import unittest
from torch import nn
from dl.block.builder.conv1dblockbuilder import Conv1DBlockBuilder


class Conv1DBlockBuilderTest(unittest.TestCase):
    def test_call(self):
        conv_1d_block_builder = Conv1DBlockBuilderTest.__create_conv_block()
        modules = conv_1d_block_builder()
        for module in modules:
            print(str(module))

    @staticmethod
    def __create_conv_block() -> Conv1DBlockBuilder:
        input_channels = 64
        output_channels = 32
        is_batch_normalization = True
        kernel_size = 3
        max_pooling_kernel = 2
        activation = nn.Tanh()
        has_bias = False
        stride = 1
        padding = 1,

        return Conv1DBlockBuilder(
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


    """
    in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool
    """
