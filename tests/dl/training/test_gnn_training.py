import unittest

from dl.block.ffnn_block import FFNNBlock
from dl.training.hyper_params import HyperParams
from dl.training.gnn_training import GNNTraining
from dl.model.gnn_base_model import GNNBaseModel
from dataset.graph_data_loader import GraphDataLoader
from torch_geometric.loader import RandomNodeLoader
from metric.metric import Metric
import torch.nn as nn
import os


class GNNTrainingTest(unittest.TestCase):
    import torch
    torch.set_default_dtype(torch.float32)

    def test_train(self):
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset = Flickr(path)
        _data = _dataset[0]
        metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.89,
            epochs=12,
            optim_label='adam',
            batch_size=4,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            encoding_len=_dataset.num_classes)

        network = GNNTraining.build(hyper_parameters, metric_labels)
        gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                               num_classes=_dataset.num_classes)
        attrs ={
            'id': 'GraphSAINTRandomWalkSampler',
            'walk_length': 2,
            'num_steps': 5,
            'batch_size': 128,
            'sample_coverage': 100
        }
        graph_data_loader = GraphDataLoader(loader_attributes=attrs,  data=_data)
        train_loader, eval_loader = graph_data_loader(num_workers=4)

        network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)

    @staticmethod
    def build(num_node_features: int, num_classes: int) -> GNNBaseModel:
        from torch_geometric.nn import GraphConv
        from dl.block.graph.gnn_base_block import GNNBaseBlock
        from dl.model.gnn_base_model import GNNBaseModel

        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gnn_block_1 = GNNBaseBlock(_id='K1',
                                   message_passing=conv_1,
                                   activation=nn.ReLU(),
                                   drop_out=0.2)
        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gnn_block_2 = GNNBaseBlock(_id='K2', message_passing=conv_2, activation=nn.ReLU(), drop_out=0.2)
        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gnn_block_3 = GNNBaseBlock(_id='K3', message_passing=conv_3, activation=nn.ReLU(), drop_out=0.2)

        ffnn_block = FFNNBlock.build(block_id='Output',
                                     layer=nn.Linear(3*hidden_channels, num_classes),
                                     activation=nn.LogSoftmax(dim=-1))
        return GNNBaseModel(model_id='Flickr',
                            gnn_blocks=[gnn_block_1, gnn_block_2, gnn_block_3],
                            ffnn_blocks=[ffnn_block])
