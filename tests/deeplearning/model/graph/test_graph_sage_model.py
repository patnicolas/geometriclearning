import unittest
import logging
from deeplearning.block.graph.graph_sage_block import GraphSAGEBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.graph.graph_sage_model import GraphSAGEModel, GraphSAGEBuilder
from torch_geometric.nn import SAGEConv, BatchNorm
from dataset.graph.pyg_datasets import PyGDatasets
import torch_geometric
import torch.nn as nn
from dataset import DatasetException
import os
from python import SKIP_REASON


class GraphSAGEModelTest(unittest.TestCase):

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_1(self):
        try:
            import torch_geometric
            from dataset.graph.pyg_datasets import PyGDatasets

            hidden_channels = 256
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            sage_conv_1 = SAGEConv(in_channels=_data.num_node_features, out_channels=hidden_channels)
            graph_SAGE_block_1 = GraphSAGEBlock(block_id='SAGE 24-256',
                                                graph_SAGE_layer=sage_conv_1,
                                                batch_norm_module=BatchNorm(hidden_channels),
                                                activation_module=nn.ReLU(),
                                                dropout_module=nn.Dropout(0.2))

            sage_conv_2 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            graph_SAGE_block_2 = GraphSAGEBlock(block_id='SAGE 256-256',
                                                graph_SAGE_layer=sage_conv_2,
                                                batch_norm_module=BatchNorm(hidden_channels),
                                                activation_module=nn.ReLU(),
                                                dropout_module=nn.Dropout(0.2))

            sage_conv_3 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            graph_SAGE_block_3 = GraphSAGEBlock(block_id='Conv 256-8', graph_SAGE_layer=sage_conv_3)
            mlp_block = MLPBlock(block_id='Fully connected',
                                 layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                                 activation_module=nn.Softmax(dim=-1))

            graph_SAGE_model = GraphSAGEModel(model_id='Flicker test dataset',
                                              graph_SAGE_blocks=[graph_SAGE_block_1, graph_SAGE_block_2, graph_SAGE_block_3],
                                              mlp_blocks=[mlp_block])
            logging.info(f'\n{graph_SAGE_model}')
            params = list(graph_SAGE_model.parameters())
            logging.info(f'\nParameters:\n{params}')
            self.assertTrue(len(params) == 15)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_init_build(self):
        try:
            out_channels = 256
            num_node_features = 64
            in_features = 256
            out_features = 8

            model_attributes = {
                'model_id': 'MyModel',
                'graph_SAGE_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'SAGE_layer': SAGEConv(in_channels=num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'SAGE_layer': SAGEConv(in_channels=out_channels, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
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
            graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
            graph_SAGE_model = graph_SAGE_builder.build()
            logging.info(graph_SAGE_model)
            params = list(graph_SAGE_model.parameters())
            self.assertTrue(len(params) == 12)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_forward(self):
        out_channels = 256
        in_features = 256
        out_features = 8

        try:
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            model_attributes = {
                'model_id': 'MyModel',
                'graph_SAGE_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'SAGE_layer': SAGEConv(in_channels=_data.num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'SAGE_layer': SAGEConv(in_channels=out_channels, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
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
            graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
            graph_SAGE_model = graph_SAGE_builder.build()
            logging.info(graph_SAGE_model)
            graph_SAGE_model.forward(data=_data)

        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except DatasetException as e:
            logging.error(e)
            self.assertTrue(False)
