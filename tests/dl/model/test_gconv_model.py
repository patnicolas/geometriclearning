import unittest

from dl.model.gnn_base_model import GNNBaseModel

from dl.block.graph.gconv_block import GConvBlock
from dl.block.mlp_block import MLPBlock
from dl.model.gconv_model import GConvModel
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling, SAGPooling
import torch.nn as nn


class GConvModelTest(unittest.TestCase):

    def test_init_1(self):
        num_node_features = 24
        num_classes = 8
        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                                   gconv_layer=conv_1,
                                   batch_norm_module=BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=nn.TopKPooling(hidden_channels, ratio=0.4),
                                   drop_out_module=nn.Dropout(0.2))

        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                                   gconv_layer=conv_2,
                                   batch_norm_module=BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=nn.TopKPooling(hidden_channels, ratio=0.4),
                                   drop_out_module=nn.Dropout(0.2))

        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_3 = GConvBlock(block_id='Conv 256-8',
                                   gconv_layer=conv_3)
        mlp_block = MLPBlock(block_id='Fully connected',
                             layer=nn.Linear(hidden_channels, num_classes),
                             activation_module=nn.Softmax())

        return GNNBaseModel.build(model_id='Flicker test dataset',
                                  gconv_blocks=[gconv_block_1, gconv_block_2, gconv_block_3])
