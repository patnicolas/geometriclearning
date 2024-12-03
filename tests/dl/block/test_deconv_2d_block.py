import unittest
from torch import nn
from dl.block.deconv_2d_block import DeConv2DBlock
from dl.block.conv_2d_block import Conv2DBlock
from dl.block.conv_block import ConvBlock
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder
from dl import ConvException
from typing import Tuple


class DeConv2DBlockTest(unittest.TestCase):

    def test_init_de_conv1(self):
        try:
            de_conv_2d_block = DeConv2DBlock( block_id='Conv_2d',
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



if __name__ == '__main__':
    unittest.main()