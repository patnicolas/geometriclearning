import unittest
import logging
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling
from deeplearning.block.graph.graph_sage_block import GraphSAGEBlock
import torch.nn as nn
import python


class GraphSAGEBlockTest(unittest.TestCase):

    def test_init_default(self):
        try:
            num_node_features = 24
            hidden_channels = 256
            graph_SAGE_layer = SAGEConv(in_channels=num_node_features, out_channels=hidden_channels)
            graph_SAGE_block = GraphSAGEBlock(block_id='SAGE-block',
                                              graph_SAGE_layer=graph_SAGE_layer,
                                              batch_norm_module=BatchNorm(hidden_channels),
                                              activation_module=nn.ReLU(),
                                              dropout_module=nn.Dropout(0.2))
            modules = list(graph_SAGE_block.modules_list)
            self.assertEqual(len(modules), 4)
            self.assertTrue(isinstance(modules[-1], nn.Dropout))
            logging.info(f'\n{graph_SAGE_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_build(self):
        try:
            num_node_features = 24
            hidden_channels = 256
            graph_SAGE_layer = SAGEConv(in_channels=num_node_features, out_channels=hidden_channels)
            block_attributes = {
                'block_id': 'My Graph SAGE block',
                'SAGE_layer': graph_SAGE_layer,
                'batch_norm': BatchNorm(hidden_channels),
                'activation': nn.ReLU(),
                'dropout': 0.3
            }
            graph_SAGE_block = GraphSAGEBlock.build(block_attributes)
            logging.info(graph_SAGE_block)
            modules = list(graph_SAGE_block.modules_list)
            self.assertEqual(len(modules), 4)
            self.assertTrue(isinstance(modules[-1], nn.Dropout))
            logging.info(f'\n{graph_SAGE_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_build_failed(self):
        from torch_geometric.nn import GraphConv

        try:
            num_node_features = 24
            hidden_channels = 256
            graph_SAGE_layer = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
            block_attributes = {
                'block_id': 'My Graph SAGE block',
                'SAGE_layer': graph_SAGE_layer,
                'batch_norm': BatchNorm(hidden_channels),
                'activation': nn.ReLU(),
                'dropout': 0.3
            }
            graph_SAGE_block = GraphSAGEBlock.build(block_attributes)
            self.assertTrue(False)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_init_build_failed_2(self):
        try:
            num_node_features = 24
            hidden_channels = 256
            graph_SAGE_layer = SAGEConv(in_channels=num_node_features, out_channels=hidden_channels)
            block_attributes = {
                'block_id': 'My Graph SAGE block',
                'SAGE_layer': graph_SAGE_layer,
                'batch_norm': BatchNorm(hidden_channels),
                'activation': nn.ReLU(),
                'dropout': 1.4
            }
            graph_SAGE_block = GraphSAGEBlock.build(block_attributes)
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
                'SAGE_layer': SAGEConv(in_channels=num_node_features, out_channels=num_channels),
                'activation': nn.ReLU(),
                'batch_norm': BatchNorm(num_channels),
                'dropout': 0.25
            }
            graph_SAGE_block = GraphSAGEBlock.build(block_attributes)

            output = graph_SAGE_block.forward(data.x, data.edge_index, data.batch)
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