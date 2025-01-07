import unittest
from torch_geometric.nn import GCNConv
from dl.block.graph.custom_gnn_block import CustomGNNBlock
import torch.nn as nn


class GCNBlockTest(unittest.TestCase):

    def test_init(self):
        from torch_geometric.datasets.karate import KarateClub

        karate_club_dataset = KarateClub()
        hidden_channels = 16
        conv = GCNConv(karate_club_dataset.num_node_features, out_channels = hidden_channels)
        gcn_conv = CustomGNNBlock('K1', conv, activation=nn.ReLU(), batch_norm=None, drop_out=0.0)
        print(repr(gcn_conv), flush=True)
        self.assertTrue(True)

    def test_init_2(self):
        from torch_geometric.datasets.karate import KarateClub

        karate_club_dataset = KarateClub()
        hidden_channels = 16
        conv = GCNConv(karate_club_dataset.num_node_features, out_channels=hidden_channels)
        gcn_conv = CustomGNNBlock('K2',
                                  conv,
                                  activation=nn.ReLU(),
                                  batch_norm=nn.BatchNorm1d(hidden_channels),
                                  drop_out=0.2)
        print(repr(gcn_conv), flush=True)
        self.assertTrue(True)

