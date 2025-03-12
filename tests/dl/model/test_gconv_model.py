import unittest

from dl.model.gnn_base_model import GNNBaseModel

from dl.block.graph.gconv_block import GConvBlock
from dl.block.mlp_block import MLPBlock
from dl.model.gconv_model import GConvModel
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling
from torch.utils.data import Dataset
import torch.nn as nn


class GConvModelTest(unittest.TestCase):

    def test_init_1(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        import torch_geometric

        hidden_channels = 256
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data: torch_geometric.data.data.Data = _dataset[0]

        conv_1 = GraphConv(in_channels=_dataset.num_node_features, out_channels=hidden_channels)
        gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                                   gconv_layer=conv_1,
                                   batch_norm_module=BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=TopKPooling(hidden_channels, ratio=0.4),
                                   dropout_module=nn.Dropout(0.2))

        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                                   gconv_layer=conv_2,
                                   batch_norm_module=BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=TopKPooling(hidden_channels, ratio=0.4),
                                   dropout_module=nn.Dropout(0.2))

        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_3 = GConvBlock(block_id='Conv 256-8',gconv_layer=conv_3)
        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                             activation_module=nn.Softmax(dim=-1))

        gconv_model = GConvModel(model_id='Flicker test dataset',
                                 gconv_blocks=[gconv_block_1, gconv_block_2, gconv_block_3],
                                 mlp_blocks=[mlp_block])
        print(f'\n{gconv_model}')
