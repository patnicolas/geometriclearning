from torch_geometric.nn.pool import TopKPooling
from dl import GNNException
import torch.nn as nn
import torch_geometric
from dataset.graph.graph_data_loader import GraphDataLoader
from dataset.graph.pyg_datasets import PyGDatasets
from torch_geometric.nn import GraphConv
from dl.block.graph.gconv_block import GConvBlock
from dl.block.mlp_block import MLPBlock
from dl.model.gconv_model import GConvModel
import torch
from torch_geometric.data import Data
from plots.plotter import PlotterParameters
from dl.training.hyper_params import HyperParams
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from typing import AnyStr, Dict, List, Any
from dl.training.gnn_training import GNNTraining
from dl.training.exec_config import ExecConfig


class EvalGConv(object):
    def __init__(self,
                 dataset_name: AnyStr,
                 hidden_channels: int,
                 dropout_p: float,
                 pooling_ratio: float,
                 lr: float,
                 epochs: int,
                 attrs: Dict[AnyStr, Any]) -> None:
        self.dataset_name = dataset_name
        self.hidden_channels = hidden_channels
        self.dropout_p = dropout_p
        self.pooling_ratio = pooling_ratio
        self.lr = lr
        self.epochs = epochs
        self.batch_size = attrs['batch_size']
        self.attrs = attrs

    def train(self):
        flickr_model, class_weights = self.__flickr_model()
        metrics, training_summary, plot_parameters, hyper_params = self.__training_env(flickr_model, class_weights)

        network = GNNTraining(hyper_params=hyper_params,
                              metrics_attributes=metrics,
                              exec_config=ExecConfig.default(),
                              plot_parameters=plot_parameters)
        train_loader, val_loader = self.__loaders()
        network.train(model_id=f'Graph Conv {self.dataset_name}',
                      neural_model=flickr_model,
                      train_loader=train_loader,
                      val_loader=val_loader)

    """ --------------------------  Private Helper Methods -----------------------  """

    def __loaders(self):
        flickr_loaders = GraphDataLoader(dataset_name='Flickr', loader_attributes=self.attrs)
        print(f'Graph data: {str(flickr_loaders.data)}')
        train_loader, val_loader = flickr_loaders()
        return train_loader, val_loader

    @staticmethod
    def __distribution(data: Data) -> torch.Tensor:
        class_distribution = data.y[data.train_mask]
        raw_distribution = torch.bincount(class_distribution)
        total_sum = raw_distribution.sum()
        distribution = raw_distribution/total_sum
        return distribution

    def __training_env(self, model: GConvModel, class_weights: torch.Tensor) -> \
            (Dict[AnyStr, BuiltInMetric], List[PlotterParameters], HyperParams):
        metric_labels = {
            MetricType.Accuracy: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
            MetricType.Precision: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
            MetricType.Recall: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
        }
        num_classes = model.mlp_blocks[-1].get_out_features()
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label.value, fig_size=(11, 7))
                      for label, _ in metric_labels.items()]

        hyper_parameters = HyperParams(
            lr=self.lr,
            momentum=0.90,
            epochs=self.epochs,
            optim_label='adam',
            batch_size=self.batch_size,
            loss_function=nn.NLLLoss(weight=class_weights.to('mps')),
            drop_out=0.2,
            train_eval_ratio=0.9,
            encoding_len=num_classes)
        return metric_labels, parameters, hyper_parameters

    def __flickr_model(self) -> (GConvModel, torch.Tensor):
        from torch_geometric.datasets.flickr import Flickr

        pyg_dataset = PyGDatasets(self.dataset_name)
        flickr_dataset: Flickr = pyg_dataset()
        if flickr_dataset is None:
            raise GNNException("Failed to load Flickr")

        _data: torch_geometric.data.Data = flickr_dataset[0]
        print(f'Number of features: {_data.num_node_features}\nNumber of classes: {flickr_dataset.num_classes}'
              f'\nSize of training: {_data.train_mask.sum()}')

        conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=self.hidden_channels)

        pooling_module = TopKPooling(self.hidden_channels, ratio=self.pooling_ratio) if self.pooling_ratio > 0 else None
        gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                                   gconv_layer=conv_1,
                                   batch_norm_module=None,  # BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=pooling_module,
                                   dropout_module=nn.Dropout(self.dropout_p))

        conv_2 = GraphConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels)
        gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                                   gconv_layer=conv_2,
                                   batch_norm_module=None,  # BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=pooling_module,
                                   dropout_module=nn.Dropout(self.dropout_p))

        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(self.hidden_channels, flickr_dataset.num_classes),
                             activation_module=nn.LogSoftmax(dim=-1))

        return GConvModel(model_id='Flicker test dataset',
                          gconv_blocks=[gconv_block_1, gconv_block_2],
                          mlp_blocks=[mlp_block]), EvalGConv.__distribution(_data)


if __name__ == '__main__':
    eval_gconv = EvalGConv(dataset_name="Flickr",
                           hidden_channels=384,
                           dropout_p=0.2,
                           pooling_ratio=-1,
                           lr=0.001,
                           epochs=96,
                           attrs={
                               'id': 'NeighborLoader',
                               'num_neighbors': [12, 6],
                               'batch_size': 512,
                               'replace': True,
                               'num_workers': 1
                           })
    eval_gconv.train()

