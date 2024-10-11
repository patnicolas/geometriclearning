import unittest
from torch import nn
from dl.block.convblock import ConvBlock
from dl.dlexception import DLException



class ConvBlockTest(unittest.TestCase):

    def test_init_conv1_succeed(self):
        dimension = 1
        out_channels = 33
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension, out_channels)
            self.assertTrue(conv_block.out_channels == out_channels)
            print(repr(conv_block))
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_conv1_failed(self):
        dimension = 1
        out_channels = 32
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension, out_channels)
            self.assertFalse(conv_block.out_channels == out_channels)
            print(repr(conv_block))
        except DLException as e:
            print(str(e))
            self.assertTrue(True)

    def test_init_conv2_succeed(self):
        dimension = 2
        out_channels = 19
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension, out_channels)
            self.assertTrue(conv_block.out_channels == out_channels)
            print(repr(conv_block))
            self.assertTrue(True)
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_conv2_failed(self):
        dimension = 2
        out_channels = 16
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension, out_channels)
            self.assertFalse(conv_block.out_channels == out_channels)
            print(repr(conv_block))
        except DLException as e:
            print(str(e))
            self.assertTrue(True)

    @staticmethod
    def __create_conv_block(dimension: int, out_channels: int) -> ConvBlock:
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        if dimension == 1:
            in_channels = 65
            kernel_size = 3
            stride = 2
            padding = 1
        elif dimension == 2:
            in_channels = 68
            kernel_size = (2, 2)
            stride = (2, 2)
            padding = (2, 2)
        else:
            raise DLException(f'Dimension {dimension} is not supported')

        has_bias = False
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