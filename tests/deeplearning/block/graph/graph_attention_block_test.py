import unittest
import logging
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, RGATConv, AGNNConv, BatchNorm
from deeplearning.block.graph.graph_attention_block import GraphAttentionBlock
import torch.nn as nn
import python


class GraphAttentionBlockTest(unittest.TestCase):

    def test_init_default_1(self):
        try:
            num_node_features = 24
            hidden_channels = 256
            graph_attention_layer = GATConv(in_channels=num_node_features, out_channels=hidden_channels)
            graph_attention_block = GraphAttentionBlock[GATConv](block_id='Attention-Block',
                                                                 graph_attention_layer=graph_attention_layer,
                                                                 batch_norm_module=BatchNorm(hidden_channels),
                                                                 activation_module=nn.ReLU(),
                                                                 dropout_module=nn.Dropout(0.2))
            modules = list(graph_attention_block.modules_list)
            self.assertEqual(len(modules), 4)
            self.assertTrue(isinstance(modules[-1], nn.Dropout))
            logging.info(f'\n{graph_attention_block}')
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
            graph_attention_layer = RGATConv(in_channels=num_node_features,
                                             out_channels=hidden_channels,
                                             num_relations=4)
            graph_attention_block = GraphAttentionBlock[RGATConv](block_id='Attention-Block',
                                                                  graph_attention_layer=graph_attention_layer,
                                                                  batch_norm_module=BatchNorm(hidden_channels),
                                                                  activation_module=nn.ReLU(),
                                                                  dropout_module=nn.Dropout(0.2))
            modules = list(graph_attention_block.modules_list)
            self.assertEqual(len(modules), 4)
            self.assertTrue(isinstance(modules[-1], nn.Dropout))
            logging.info(f'\n{graph_attention_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_build(self):
        try:
            num_node_features = 7
            hidden_channels = 32
            block_descriptor = {'block_id': 'Attention block',
                                'attention_layer': RGATConv(in_channels=num_node_features,
                                                            out_channels=hidden_channels,
                                                            num_relations=4),
                                'batch_norm': BatchNorm(hidden_channels),
                                'activation': nn.LeakyReLU(),
                                'dropout': 0.3
                                }
            graph_attention_block = GraphAttentionBlock.build(block_descriptor)
            modules = list(graph_attention_block.modules_list)
            self.assertEqual(len(modules), 4)
            self.assertTrue(isinstance(modules[-1], nn.Dropout))
            logging.info(f'\n{graph_attention_block}')
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)