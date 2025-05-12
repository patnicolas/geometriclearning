import unittest
from torch_geometric.nn import GCNConv
from dl.block.graph.g_message_passing_block import GMessagePassingBlock
import torch.nn as nn
import logging

class GMessagePassingBlockTest(unittest.TestCase):

    def test_init(self):
        from torch_geometric.datasets.karate import KarateClub

        karate_club_dataset = KarateClub()
        hidden_channels = 16
        conv = GCNConv(karate_club_dataset.num_node_features, out_channels = hidden_channels)
        gcn_conv = GMessagePassingBlock('K1', conv, activation_module=nn.ReLU(), batch_norm_module=None, drop_out_module=0.0)
        logging.info(repr(gcn_conv))
        self.assertTrue(True)

    def test_init_2(self):
        from torch_geometric.datasets.karate import KarateClub
        from torch_geometric.nn import BatchNorm

        karate_club_dataset = KarateClub()
        hidden_channels = 16
        conv = GCNConv(karate_club_dataset.num_node_features, out_channels=hidden_channels)
        gcn_conv = GMessagePassingBlock('K2',
                                        conv,
                                        activation_module=nn.ReLU(),
                                        batch_norm_module=BatchNorm(hidden_channels),
                                        dropout_module=0.2)
        logging.info(repr(gcn_conv), flush=True)
        self.assertTrue(True)

