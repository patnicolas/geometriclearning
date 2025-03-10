
import unittest
from torch_geometric.nn import GraphConv
from torch_geometric.nn import BatchNorm
from dl.block.graph.gconv_block import GConvBlock
import torch.nn as nn

class GConvBlockTest(unittest.TestCase):

    def test_init_1(self):
        num_node_features = 24
        hidden_channels = 256
        gconv_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gconv_block = GConvBlock(block_id='GConv-block',
                                 gconv_layer=gconv_layer,
                                 batch_norm_module=BatchNorm(hidden_channels),
                                 activation_module=nn.ReLU(),
                                 pooling_module=nn.TopKPooling(hidden_channels, ratio=0.4),
                                 drop_out_p=nn.Dropout(0.2))
        self.assertTrue(len(gconv_block.modules) == 4)
        print(f'\n{gconv_block}')

    def test_init_2(self):
        num_node_features = 24
        hidden_channels = 256
        gconv_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gconv_block = GConvBlock(block_id='GConv-block',
                                 gconv_layer=gconv_layer,
                                 batch_norm_module=BatchNorm(hidden_channels))
        self.assertTrue(len(gconv_block.modules) == 2)
        print(f'\n{gconv_block}')


