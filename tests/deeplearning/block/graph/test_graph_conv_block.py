import unittest
import logging
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling
from deeplearning.block.graph.graph_conv_block import GraphConvBlock
import torch.nn as nn
import python

class GraphConvBlockTest(unittest.TestCase):

    def test_init_default(self):
        try:
            num_node_features = 24
            hidden_channels = 256
            graph_conv_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
            graph_conv_block = GraphConvBlock(block_id='GConv-block',
                                              graph_conv_layer=graph_conv_layer,
                                              batch_norm_module=BatchNorm(hidden_channels),
                                              activation_module=nn.ReLU(),
                                              pooling_module=TopKPooling(hidden_channels, ratio=0.4),
                                              dropout_module=nn.Dropout(0.2))
            modules = list(graph_conv_block.modules_list)
            self.assertEqual(len(modules), 5)
            self.assertTrue(isinstance(modules[-1], nn.Dropout))
            logging.info(f'\n{graph_conv_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_init_default_2(self):
        try:
            num_node_features = 24
            hidden_channels = 256
            graph_conv_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
            graph_conv_block = GraphConvBlock(block_id='GConv-block',
                                              graph_conv_layer=graph_conv_layer,
                                              batch_norm_module=BatchNorm(hidden_channels))
            modules = list(graph_conv_block.modules_list)
            self.assertEqual(len(modules), 2)
            logging.info(f'\n{graph_conv_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_build(self):
        try:
            num_node_features = 24
            num_channels = 256
            block_attributes = {
                'block_id': 'MyBlock',
                'conv_layer': GraphConv(in_channels=num_node_features, out_channels=num_channels),
                'activation': nn.ReLU(),
                'batch_norm': BatchNorm(num_channels),
                'pooling': None,
                'dropout': 0.25
            }
            graph_conv_block = GraphConvBlock.build(block_attributes)
            modules = list(graph_conv_block.modules_list)

            self.assertEqual(len(modules), 4)
            logging.info(f'\n{graph_conv_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_build_failed(self):
        from torch_geometric.nn import SAGEConv
        try:
            num_node_features = 24
            num_channels = 256
            block_attributes = {
                'block_id': 'MyBlock',
                'conv_layer': SAGEConv(in_channels=num_node_features, out_channels=num_channels),
                'activation': nn.ReLU(),
                'batch_norm': BatchNorm(num_channels),
                'pooling': None,
                'dropout': 0.25
            }
            graph_conv_block = GraphConvBlock.build(block_attributes)
            self.assertTrue(False)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_forward(self):
        try:
            from dataset.graph.pyg_datasets import PyGDatasets

            dataset_name = 'Flickr'
            pyg_datasets = PyGDatasets(dataset_name)
            dataset = pyg_datasets()
            data = dataset[0]
            logging.info(f'\nInput data:\n{str(data)}')

            num_node_features = data.num_features
            num_channels = 256
            block_attributes = {
                'block_id': 'MyBlock',
                'conv_layer': GraphConv(in_channels=num_node_features, out_channels=num_channels),
                'activation': nn.ReLU(),
                'batch_norm': BatchNorm(num_channels),
                'pooling': None,
                'dropout': 0.25
            }
            graph_conv_block = GraphConvBlock.build(block_attributes)

            output = graph_conv_block.forward(data.x, data.edge_index, data.batch)
            logging.info(output)
            self.assertEqual(output.shape[1], 256)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except RuntimeError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_forward_failed(self):
        try:
            from dataset.graph.pyg_datasets import PyGDatasets

            dataset_name = 'Flickr'
            pyg_datasets = PyGDatasets(dataset_name)
            dataset = pyg_datasets()
            data = dataset[0]
            logging.info(f'\nInput data:\n{str(data)}')

            num_node_features = 4 # ata.num_features
            num_channels = 256
            block_attributes = {
                'block_id': 'MyBlock',
                'conv_layer': GraphConv(in_channels=num_node_features, out_channels=num_channels),
                'activation': nn.ReLU(),
                'batch_norm': BatchNorm(num_channels),
                'pooling': None,
                'dropout': 0.25
            }
            graph_conv_block = GraphConvBlock.build(block_attributes)

            output = graph_conv_block.forward(data.x, data.edge_index, data.batch)
            logging.info(output)
            self.assertTrue(False)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)
        except RuntimeError as e:
            logging.error(e)
            self.assertTrue(True)



