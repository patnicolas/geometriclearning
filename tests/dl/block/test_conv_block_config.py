import unittest
from dl.block.conv_block_config import ConvBlockConfig
import torch.nn as nn


class ConvBlockConfigTest(unittest.TestCase):
    def test_invert(self):
        conv_block_config = ConvBlockConfig(in_channels=8,
                                            out_channels=32,
                                            kernel_size=(2, 2),
                                            stride=(2, 2),
                                            padding=(2, 2),
                                            batch_norm=True,
                                            max_pooling_kernel=2,
                                            activation=nn.Tanh(),
                                            bias=False)
        conv_block_config.transpose()
        self.assertTrue(conv_block_config.out_channels == 8)
        self.assertTrue(conv_block_config.in_channels == 32)