import unittest
from torch import nn
from dl.block.convblock import ConvBlock

class ConvBlockTest(unittest.TestCase):

    def test_init_conv1(self):
        dimension = 1
        conv_block = ConvBlockTest.__create_conv_block(dimension)
        print(repr(conv_block))

    def test_init_conv2(self):
        dimension = 2
        conv_block = ConvBlockTest.__create_conv_block(dimension)
        print(repr(conv_block))

    def test_module_weights(self):
        dimension = 2
        conv_block = ConvBlockTest.__create_conv_block(dimension)
        weights = conv_block.get_modules_weights()
        print(f'Weights: {weights}')

    @staticmethod
    def __create_conv_block(dimension: int) -> ConvBlock:
        in_channels = 64
        out_channels = 32
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        kernel_size = 3
        has_bias = False
        stride = 1
        padding = 1
        return ConvBlock(
            dimension,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            bias=has_bias)


if __name__ == '__main__':
    unittest.main()