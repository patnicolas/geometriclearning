import unittest

from dataset import DatasetException
from dataset.graph.graph_data_loader import GraphDataLoader
from dl.training.gnn_training import GNNTraining
from dl.training.exec_config import ExecConfig
from dl.training.training_summary import TrainingSummary
from plots.plotter import PlotterParameters
from dl import GNNException, DLException, TrainingException
from dl.model.gconv_model import GConvModel
from dl.training.hyper_params import HyperParams
from metric.metric import Metric
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from torch_geometric.data import Data
import torch.nn as nn
import torch


class GConvTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init_model(self):
        hidden_channels = 256
        pooling_ratio = 0.4
        dropout_p = 0.2

        try:
            flickr_model = GConvTest.flickr_model(hidden_channels, pooling_ratio, dropout_p)
            print(flickr_model)
            self.assertTrue(True)
        except GNNException as e:
            print(e)
            self.assertTrue(False)

    def test_training_test(self):
        hidden_channels = 256
        pooling_ratio = 0.4
        dropout_p = 0.2

        try:
            flickr_model, class_weights = GConvTest.flickr_model(hidden_channels, pooling_ratio, dropout_p)
            metric_labels = {
                Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
                Metric.precision_label: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
                Metric.recall_label: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
            }
            num_classes = flickr_model.mlp_blocks[-1].get_out_features()
            training_summary = TrainingSummary(patience=2, min_diff_loss=-0.002, early_stopping_enabled=True)
            parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                          for label, _ in metric_labels.items()]

            hyper_parameters = HyperParams(
                lr=0.001,
                momentum=0.90,
                epochs=64,
                optim_label='adam',
                batch_size=512,
                loss_function=nn.NLLLoss(weight=class_weights.to('mps')),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=num_classes)

            attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': [24, 12, 4],
                'batch_size': 512,
                'replace': True,
                'num_workers': 1
            }
            flickr_loaders = GraphDataLoader(attrs, 'Flickr')
            print(f'Graph data: {str(flickr_loaders.data)}')
            train_loader, val_loader = flickr_loaders()

            network = GNNTraining(hyper_params=hyper_parameters,
                                  training_summary=training_summary,
                                  metrics=metric_labels,
                                  exec_config=ExecConfig.default(),
                                  plot_parameters=parameters)
            network.train(model_id='Graph Conv Flickr',
                          neural_model=flickr_model,
                          train_loader=train_loader,
                          val_loader=val_loader)
        except (GNNException | DLException | DatasetException | TrainingException) as e:
            print(e)
            self.assertTrue(False)

    @staticmethod
    def flickr_model(hidden_channels: int, pooling_ratio: float, dropout_p: float) -> (GConvModel, torch.Tensor):
        import torch_geometric
        from dataset.graph.pyg_datasets import PyGDatasets
        from torch_geometric.nn import GraphConv
        from dl.block.graph.gconv_block import GConvBlock
        from dl.block.mlp_block import MLPBlock
        from torch_geometric.datasets.flickr import Flickr

        pyg_dataset = PyGDatasets('Flickr')
        flickr_dataset: Flickr = pyg_dataset()
        if flickr_dataset is None:
            raise GNNException("Failed to load Flickr")

        _data: torch_geometric.data.Data = flickr_dataset[0]
        print(f'Number of features: {_data.num_node_features}\nNumber of classes: {flickr_dataset.num_classes}'
              f'\nSize of training: {_data.train_mask.sum()}')

        conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels)

        gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                                   gconv_layer=conv_1,
                                   batch_norm_module=None, #BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=None,
                                   dropout_module=nn.Dropout(dropout_p))

        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                                   gconv_layer=conv_2,
                                   batch_norm_module=None, #BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=None, #TopKPooling(hidden_channels, ratio=pooling_ratio))
                                   dropout_module=nn.Dropout(dropout_p))

        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_3 = GConvBlock(block_id='Conv 256-8', gconv_layer=conv_3)
        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(hidden_channels, flickr_dataset.num_classes),
                             activation_module=nn.LogSoftmax(dim=-1))

        return GConvModel(model_id='Flicker test dataset',
                          gconv_blocks=[gconv_block_1, gconv_block_2],
                          mlp_blocks=[mlp_block]), GConvTest.distribution(_data)

    @staticmethod
    def distribution(data: Data) -> torch.Tensor:
        class_distribution = data.y[data.train_mask]
        raw_distribution = torch.bincount(class_distribution)
        s = raw_distribution.sum()
        distribution = raw_distribution /s
        return distribution



