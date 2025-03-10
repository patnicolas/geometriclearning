import unittest
from torch import nn

from dl.block.conv.conv_block_config import ConvBlockConfig
from dl.block.conv.deconv_2d_block import DeConv2dBlock
from dl import ConvException


class DeConv2DBlockTest(unittest.TestCase):

    def test_init_de_conv1(self):
        try:
            de_conv_2d_block = DeConv2dBlock.build(block_id='Conv_2d',
                                                   in_channels=16,
                                                   out_channels=10,
                                                   kernel_size=(3, 3),
                                                   stride=(1, 1),
                                                   padding=(0, 0),
                                                   batch_norm=True,
                                                   activation=nn.ReLU(),
                                                   bias=False)
            print(repr(de_conv_2d_block))
            self.assertTrue(de_conv_2d_block.conv_block_config.out_channels == 10)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_de_conv2(self):
        try:
            conv_block_config = ConvBlockConfig.de_conv(in_channels=16,
                                                        out_channels=10,
                                                        kernel_size=(3, 3),
                                                        stride=(1, 1),
                                                        padding=(0, 0),
                                                        batch_norm=True,
                                                        activation=nn.ReLU(),
                                                        bias=False)
            de_conv_2d_block = DeConv2dBlock(block_id='De_Conv',
                                             conv_block_config=conv_block_config,
                                             activation=nn.Sigmoid())
            print(repr(de_conv_2d_block))
            self.assertTrue(str(de_conv_2d_block.conv_block_config.activation_module) == 'Sigmoid()')
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()