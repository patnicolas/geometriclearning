import unittest
import logging

from dl.block import GraphException
from dl.block.graph.gconv_block import GConvBlock
from dl.block.mlp_block import MLPBlock
from dl.model.gconv_model import GConvModel
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling
import torch.nn as nn
import os
import python
from python import SKIP_REASON


class GConvModelTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_1(self):
        import torch_geometric
        from dataset.graph.pyg_datasets import PyGDatasets

        try:
            hidden_channels = 256
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels)
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
            logging.info(f'\n{gconv_model}')
            params = list(gconv_model.parameters())
            logging.info(f'\nParameters:\n{params}')
            self.assertTrue(len(params) == 17)
        except (AssertionError | GraphException) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_2(self):
        out_channels = 256
        num_node_features = 64
        in_features = 256
        out_features = 8

        model_attributes = {
            'model_id': 'MyModel',
            'gconv_blocks': [
                {
                    'block_id': 'MyBlock_1',
                    'conv_layer': GraphConv(in_channels=num_node_features, out_channels=out_channels),
                    'num_channels': out_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': BatchNorm(out_channels),
                    'pooling': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'MyBlock_2',
                    'conv_layer': GraphConv(in_channels=out_channels, out_channels=out_channels),
                    'num_channels': out_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': BatchNorm(out_channels),
                    'pooling': None,
                    'dropout': 0.25
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'MyMLP',
                    'in_features': in_features,
                    'out_features': out_features,
                    'activation': nn.ReLU(),
                    'dropout': 0.3
                }
            ]
        }
        try:
            gconv_model = GConvModel.build(model_attributes)
            logging.info(gconv_model)
        except (AssertionError | KeyError | GraphException) as e:
            logging.error(e)
            self.assertTrue(False)
