import unittest
from torch import nn
from dl.block.conv_block import ConvBlock
from dl import ConvException
from dl.block.builder.conv1d_block_builder import Conv1DBlockBuilder
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder


class ConvBlockTest(unittest.TestCase):

    def test_init_conv1(self):
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension=1, out_channels=33)
            self.assertTrue(conv_block.conv_block_config.out_channels == 33)
            print(str(conv_block))
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_conv2(self):
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension=2, out_channels=19)
            self.assertTrue(conv_block.conv_block_config.out_channels == 19)
            print(repr(conv_block))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @staticmethod
    def __create_conv_block(dimension: int, out_channels: int) -> ConvBlock:
        from dl.block.conv_block import ConvBlockConfig
        if dimension == 2:
            conv_block_config = ConvBlockConfig(in_channels=68,
                                                out_channels=out_channels,
                                                kernel_size=(2,2),
                                                stride=(2, 2),
                                                padding=(2, 2),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.Tanh(),
                                                bias=False)
            return ConvBlock(block_id='Conv_2d', conv_block_config=conv_block_config, modules=(nn.Sigmoid(),))
        elif  dimension == 1:
            conv_block_config = ConvBlockConfig(in_channels=65,
                                                out_channels=out_channels,
                                                kernel_size=3,
                                                stride=3,
                                                padding=1,
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.Tanh(),
                                                bias=False)
            return ConvBlock(block_id='Conv_1d', conv_block_config=conv_block_config, modules=(nn.Sigmoid(),))
        else:
            raise ConvException(f'Dimension {dimension} is not supported')


if __name__ == '__main__':
    unittest.main()