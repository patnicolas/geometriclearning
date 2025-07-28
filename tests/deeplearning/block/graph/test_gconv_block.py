import unittest
import logging
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling
from deeplearning.block.graph.gconv_block import GConvBlock
import torch.nn as nn
import python

class GConvBlockTest(unittest.TestCase):

    def test_init_1(self):
        num_node_features = 24
        hidden_channels = 256
        gconv_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gconv_block = GConvBlock(block_id='GConv-block',
                                 gconv_layer=gconv_layer,
                                 batch_norm_module=BatchNorm(hidden_channels),
                                 activation_module=nn.ReLU(),
                                 pooling_module=TopKPooling(hidden_channels, ratio=0.4),
                                 dropout_module=nn.Dropout(0.2))
        modules = list(gconv_block.modules_list)
        self.assertTrue(len(modules) == 5)
        logging.info(f'\n{gconv_block}')

    def test_init_2(self):
        num_node_features = 24
        hidden_channels = 256
        gconv_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gconv_block = GConvBlock(block_id='GConv-block',
                                 gconv_layer=gconv_layer,
                                 batch_norm_module=BatchNorm(hidden_channels))
        modules = list(gconv_block.modules_list)
        self.assertTrue(len(modules) == 2)
        logging.info(f'\n{gconv_block}')

    def test_init_3(self):
        num_node_features = 24
        num_channels = 256
        block_attributes = {
            'block_id': 'MyBlock',
            'conv_layer': GraphConv(in_channels=num_node_features, out_channels=num_channels),
            'num_channels': num_channels,
            'activation': nn.ReLU(),
            'batch_norm': BatchNorm(num_channels),
            'pooling': None,
            'dropout': 0.25
        }
        gconv_block = GConvBlock.build(block_attributes)
        modules = list(gconv_block.modules_list)
        self.assertTrue(len(modules) == 4)
        logging.info(f'\n{gconv_block}')


