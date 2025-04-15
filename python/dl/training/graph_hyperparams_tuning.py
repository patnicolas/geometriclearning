__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import optuna
from optuna.trial import TrialState
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
from dl.training.training_summary import TrainingSummary
from plots.plotter import PlotterParameters
from dl.training.hyper_params import HyperParams
from metric.metric import Metric
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from typing import AnyStr, Dict, List
from dl.training.gnn_training import GNNTraining
from torch.utils.data import DataLoader


def distribution(data: Data) -> torch.Tensor:
    class_distribution = data.y[data.train_mask]
    raw_distribution = torch.bincount(class_distribution)
    total_sum = raw_distribution.sum()
    return raw_distribution / total_sum


def flickr_model(dataset_name, hidden_channels, pooling_ratio, dropout_p) -> (GConvModel, torch.Tensor):
    from torch_geometric.datasets.flickr import Flickr

    pyg_dataset = PyGDatasets(dataset_name)
    flickr_dataset: Flickr = pyg_dataset()
    if flickr_dataset is None:
        raise GNNException("Failed to load Flickr")

    _data: torch_geometric.data.Data = flickr_dataset[0]
    print(f'Number of features: {_data.num_node_features}\nNumber of classes: {flickr_dataset.num_classes}'
          f'\nSize of training: {_data.train_mask.sum()}')

    conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels)

    pooling_module = TopKPooling(hidden_channels, ratio=pooling_ratio) if pooling_ratio > 0 else None
    gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                               gconv_layer=conv_1,
                               batch_norm_module=None,  # BatchNorm(hidden_channels),
                               activation_module=nn.ReLU(),
                               pooling_module=pooling_module,
                               dropout_module=nn.Dropout(dropout_p))

    conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
    gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                               gconv_layer=conv_2,
                               batch_norm_module=None,  # BatchNorm(hidden_channels),
                               activation_module=nn.ReLU(),
                               pooling_module=pooling_module,
                               dropout_module=nn.Dropout(dropout_p))

    mlp_block = MLPBlock(block_id='Fully connected',
                         layer_module=nn.Linear(hidden_channels, flickr_dataset.num_classes),
                         activation_module=nn.LogSoftmax(dim=-1))

    return GConvModel(model_id='Flicker test dataset',
                      gconv_blocks=[gconv_block_1, gconv_block_2],
                      mlp_blocks=[mlp_block]), distribution(_data)


def define_model(trial) -> (GConvModel, torch.Tensor):
    gconv_model, class_weights = flickr_model(dataset_name='Flickr',
                                              hidden_channels=512,
                                              pooling_ratio=-1,
                                              dropout_p=0.2)
    return gconv_model, class_weights

def training_env(trail, model: GConvModel, class_weights: torch.Tensor) -> \
            (Dict[AnyStr, BuiltInMetric], TrainingSummary, List[PlotterParameters], HyperParams):
    metric_labels = {
        Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
        Metric.precision_label: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
        Metric.recall_label: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
    }
    num_classes = model.mlp_blocks[-1].get_out_features()
    training_summary = TrainingSummary(patience=2, min_diff_loss=-0.002, early_stopping_enabled=True)
    parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                  for label, _ in metric_labels.items()]

    lr = trail.suggest_categorical('lr', [0.01, 0.002, 0.0005])
    hyper_parameters = HyperParams(lr=lr,
                                   momentum=0.90,
                                   epochs=64,
                                   optim_label='adam',
                                   batch_size=512,
                                   loss_function=nn.NLLLoss(weight=class_weights.to('mps')),
                                   drop_out=0.2,
                                   train_eval_ratio=0.9,
                                   encoding_len=num_classes)
    return metric_labels, training_summary, parameters, hyper_parameters

def loaders(num_neighbors: int, num_hops: int, batch_size: int) -> (DataLoader, DataLoader):
    flickr_loaders = GraphDataLoader.build_neighbor_loader(
        dataset_name='Flickr',
        n_neighbors=neighbor_list(num_neighbors, num_hops),
        b_size=batch_size,
        num_workers=4)
    print(f'Graph data: {str(flickr_loaders.data)}')
    train_loader, val_loader = flickr_loaders()
    return train_loader, val_loader

def neighbor_list(num_neighbors_1: int, num_hops: int) -> List[int]:
    if num_hops > 2:
        return [num_neighbors_1, num_neighbors_1 // 2, num_neighbors_1 // 4]
    elif num_hops > 1:
        return [num_neighbors_1, num_neighbors_1 // 2]
    else:
        return [num_neighbors_1]

def objective(trial) -> float:
    _model, class_weights = define_model(trial)
    metrics, training_summary, _, hyper_params = training_env(trial, _model, class_weights)

    network = GNNTraining(hyper_params=hyper_params,
                          training_summary=training_summary,
                          metrics=metrics)
    num_neighbors_1 = trial.suggest_categorical('num_neighbors_1', [4, 8, 12, 24])
    num_hops = trial.suggest_categorical('num_hops', [2, 3])
    batch_size = trial.suggest_categorical('batch_size', [128, 512])
    train_loader, val_loader = loaders(num_neighbors_1, num_hops, batch_size)
    output_id = f'Flickr_n_{num_neighbors_1}_h_{num_hops}_b_{batch_size}_lr_{hyper_params.learning_rate}'
    network.train(output_id, _model, train_loader, val_loader)
    accuracy = float(training_summary.metrics[Metric.accuracy_label][-1])
    print('Accuracy')
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(study_name='Flickr', direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


