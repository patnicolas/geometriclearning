import unittest

from dl.block.conv_2d_block import Conv2DBlock
import torch.nn as nn


class Conv2DBlockTest(unittest.TestCase):
    def test_init_1(self):
        conv_2d_block = Conv2DBlock(block_id='Conv_2d',
                                    in_channels=1,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    batch_norm=True,
                                    max_pooling_kernel=2,
                                    activation=nn.ReLU(),
                                    bias=False)
        print(str(conv_2d_block))
        self.assertTrue(conv_2d_block.conv_block_config.out_channels == 32)

    def test_invert(self):
        conv_2d_block = Conv2DBlock(block_id='Conv_2d',
                                    in_channels=1,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    batch_norm=True,
                                    max_pooling_kernel=2,
                                    activation=nn.ReLU(),
                                    bias=False)
        de_conv_2d_block = conv_2d_block.invert(extra=nn.Sigmoid())
        print(str(de_conv_2d_block))