import unittest

from dataset import DatasetException
from dataset.graph.graph_data_loader import GraphDataLoader
from dl.training.gnn_training import GNNTraining
from dl.training.exec_config import ExecConfig
from plots.plotter import PlotterParameters
from dl import MLPException, TrainingException
from dl.block.graph import GraphException
from dl.model.gconv_model import GConvModel
from dl.training.hyper_params import HyperParams
from metric.metric import Metric
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from torch_geometric.data import Data
import torch.nn as nn
import torch
import logging
import os
import python
from python import SKIP_REASON


class GConvTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualization_3d(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        tensor = torch.rand(20, 10, 10)  # [D, H, W]

        fig, ax = plt.subplots()
        im = ax.imshow(tensor[0], cmap='viridis')

        def update(i):
            im.set_data(tensor[i])
            ax.set_title(f'Slice {i}')

        ani = FuncAnimation(fig, update, frames=range(tensor.size(0)), interval=300)
        plt.show()

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualization_3d2(self):
        import torch
        import matplotlib.pyplot as plt

        tensor = torch.rand(5, 10, 10)  # shape: [C, H, W]
        fig, axs = plt.subplots(1, tensor.size(0), figsize=(15, 3))
        for i in range(tensor.size(0)):
            axs[i].imshow(tensor[i].numpy(), cmap='viridis')
            axs[i].set_title(f'Slice {i}')
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_model(self):
        hidden_channels = 256
        pooling_ratio = 0.4
        dropout_p = 0.2

        try:
            flickr_model = GConvTest.flickr_model(hidden_channels, dropout_p)
            logging.info(flickr_model)
            self.assertTrue(True)
        except GraphException as e:
            logging.info(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_training_test(self):
        hidden_channels = 256
        pooling_ratio = 0.4
        dropout_p = 0.2
        target_device = 'mps'

        try:
            flickr_model, class_weights = GConvTest.flickr_model(hidden_channels, pooling_ratio, dropout_p)
            metric_labels = {
                Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
                Metric.precision_label: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
                Metric.recall_label: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
            }
            num_classes = flickr_model.mlp_blocks[-1].get_out_features()
            parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                          for label, _ in metric_labels.items()]

            hyper_parameters = HyperParams(
                lr=0.001,
                momentum=0.90,
                epochs=64,
                optim_label='adam',
                batch_size=512,
                loss_function=nn.NLLLoss(weight=class_weights.to(target_device)),
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
            flickr_loaders = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            logging.info(f'Graph data: {str(flickr_loaders.data)}')
            train_loader, val_loader = flickr_loaders()

            network = GNNTraining(hyper_params=hyper_parameters,
                                  metrics_attributes=metric_labels,
                                  exec_config=ExecConfig.default(),
                                  plot_parameters=parameters)
            network.train(model_id='Graph Conv Flickr',
                          neural_model=flickr_model,
                          train_loader=train_loader,
                          val_loader=val_loader)
        except (GraphException | MLPException | DatasetException | TrainingException) as e:
            logging.info(e)
            self.assertTrue(False)

    @staticmethod
    def flickr_model(hidden_channels: int, dropout_p: float) -> (GConvModel, torch.Tensor):
        import torch_geometric
        from dataset.graph.pyg_datasets import PyGDatasets
        from torch_geometric.nn import GraphConv
        from dl.block.graph.gconv_block import GConvBlock
        from dl.block.mlp_block import MLPBlock
        from torch_geometric.datasets.flickr import Flickr

        pyg_dataset = PyGDatasets('Flickr')
        flickr_dataset: Flickr = pyg_dataset()
        if flickr_dataset is None:
            raise GraphException("Failed to load Flickr")

        _data: torch_geometric.data.Data = flickr_dataset[0]
        logging.info(f'Number of features: {_data.num_node_features}\nNumber of classes: {flickr_dataset.num_classes}'
                     f'\nSize of training: {_data.train_mask.sum()}')

        conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels)

        gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                                   gconv_layer=conv_1,
                                   batch_norm_module=None,
                                   activation_module=nn.ReLU(),
                                   pooling_module=None,
                                   dropout_module=nn.Dropout(dropout_p))

        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                                   gconv_layer=conv_2,
                                   batch_norm_module=None,
                                   activation_module=nn.ReLU(),
                                   pooling_module=None,
                                   dropout_module=nn.Dropout(dropout_p))

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



