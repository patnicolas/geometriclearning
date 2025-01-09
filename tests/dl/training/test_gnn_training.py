import unittest

from dl.training.hyper_params import HyperParams
from dl.training.gnn_training import GNNTraining
from dl.model.gnn_base_model import GNNBaseModel
from torch_geometric.loader import NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler
from metric.metric import Metric
import torch.nn as nn
import os

class GNNTrainingTest(unittest.TestCase):

    def test_train(self):
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        lr = 0.001
        hyper_parameters = HyperParams(
            lr=lr,
            momentum=0.89,
            epochs=2,
            optim_label='adam',
            batch_size=4,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            encoding_len=101)

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        data = _dataset[0]
        metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]

        network = GNNTraining.build(hyper_parameters, metric_labels)
        gcn_model = GNNTrainingTest.build(batch_size=4,
                                          walk_length=6,
                                          num_node_features=_dataset[0].num_nodes,
                                          num_classes=_dataset[0].num_classes)
        network.train(gcn_model.model_id,
                      gcn_model.model,
                      RandomNodeLoader(data, num_parts=3),
                      RandomNodeLoader(data, num_parts=3))

    @staticmethod
    def build(batch_size: int, walk_length: int, num_node_features: int, num_classes: int) -> GNNBaseModel:
        from torch_geometric.nn import BatchNorm, GraphConv
        from dl.block.graph.gnn_base_block import GNNBaseBlock
        from dl.model.gnn_base_model import GNNBaseModel

        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gcn_conv_1 = GNNBaseBlock(_id='K1',
                                  message_passing=conv_1,
                                  activation=nn.ReLU(),
                                  batch_norm=BatchNorm(hidden_channels),
                                  drop_out=0.2)
        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gcn_conv_2 = GNNBaseBlock(_id='K2', message_passing=conv_2, activation=nn.ReLU())
        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=num_classes)
        gcn_conv_3 = GNNBaseBlock(_id='K3', message_passing=conv_3)

        return GNNBaseModel.build(model_id='Flicker',
                                  batch_size=batch_size,
                                  walk_length=walk_length,
                                  gcn_blocks=[gcn_conv_1, gcn_conv_2, gcn_conv_3])
