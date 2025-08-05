__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import Dict, List, AnyStr, Any
import logging
# 3rd Party imports
import optuna
from optuna.trial import TrialState
from torch_geometric.nn.pool import TopKPooling
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphConv
# Library imports
from dataset.graph.graph_data_loader import GraphDataLoader
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.block.graph.graph_conv_block import GraphConvBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.graph.graph_conv_model import GraphConvModel
from plots.plotter import PlotterParameters
from deeplearning.training.hyper_params import HyperParams
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from deeplearning.block.graph import GraphException
from deeplearning.training.gnn_training import GNNTraining
__all__ = ['distribution', 'flickr_model', 'training_env', 'loaders', 'init_parameters_optimizer', 'objective']


def distribution(data: Data) -> torch.Tensor:
    class_distribution = data.y[data.train_mask]
    raw_distribution = torch.bincount(class_distribution)
    total_sum = raw_distribution.sum()
    return raw_distribution / total_sum


def flickr_model(dataset_name, hidden_channels, pooling_ratio, dropout_p) -> (GraphConvModel, torch.Tensor):
    from torch_geometric.datasets.flickr import Flickr

    pyg_dataset = PyGDatasets(dataset_name)
    flickr_dataset: Flickr = pyg_dataset()
    if flickr_dataset is None:
        raise GraphException("Failed to load Flickr")

    _data: torch_geometric.data.Data = flickr_dataset[0]
    logging.info(f'Number of features: {_data.num_node_features}\nNumber of classes: {flickr_dataset.num_classes}'
          f'\nSize of training: {_data.train_mask.sum()}')

    conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels)

    pooling_module = TopKPooling(hidden_channels, ratio=pooling_ratio) if pooling_ratio > 0 else None
    gconv_block_1 = GraphConvBlock(block_id='Conv 24-256',
                                   graph_conv_layer=conv_1,
                                   batch_norm_module=None,  # BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=pooling_module,
                                   dropout_module=nn.Dropout(dropout_p))

    conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
    gconv_block_2 = GraphConvBlock(block_id='Conv 256-256',
                                   graph_conv_layer=conv_2,
                                   batch_norm_module=None,  # BatchNorm(hidden_channels),
                                   activation_module=nn.ReLU(),
                                   pooling_module=pooling_module,
                                   dropout_module=nn.Dropout(dropout_p))

    mlp_block = MLPBlock(block_id='Fully connected',
                         layer_module=nn.Linear(hidden_channels, flickr_dataset.num_classes),
                         activation_module=nn.LogSoftmax(dim=-1))

    return GraphConvModel(model_id='Flicker test dataset',
                          graph_conv_blocks=[gconv_block_1, gconv_block_2],
                          mlp_blocks=[mlp_block]), distribution(_data)


def training_env(model: GraphConvModel, class_weights: torch.Tensor) -> GNNTraining:
    metric_labels = {
        MetricType.Accuracy: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
        MetricType.Precision: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
        MetricType.Recall: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
    }
    num_classes = model.mlp_blocks[-1].get_out_features()
    parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label.value, fig_size=(11, 7))
                  for label, _ in metric_labels.items()]

    # lr = trail.suggest_categorical('lr', [0.01, 0.002, 0.0005])
    hyper_parameters = HyperParams(lr=0.02,
                                   momentum=0.90,
                                   epochs=64,
                                   optim_label='adam',
                                   batch_size=512,
                                   loss_function=nn.NLLLoss(weight=class_weights.to('mps')),
                                   drop_out=0.2,
                                   train_eval_ratio=0.9,
                                   encoding_len=num_classes)
    # return metric_labels, parameters, hyper_parameters
    return GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels, plot_parameters=parameters)

def loaders(graph_attributes: Dict[AnyStr, Any]) -> (DataLoader, DataLoader):
    flickr_loaders = GraphDataLoader(dataset_name='Flickr', sampling_attributes=graph_attributes)
    logging.info(f'Graph data: {str(flickr_loaders.data)}')
    train_loader, val_loader = flickr_loaders()
    return train_loader, val_loader

def init_parameters_optimizer(trial) -> Dict[AnyStr, Any]:
    num_neighbors_hop_1 = trial.suggest_categorical('num_neighbors_1', [4, 8, 12, 24])
    num_hops = trial.suggest_categorical('num_hops', [2, 3])
    batch_size = trial.suggest_categorical('batch_size', [128, 512])

    def neighbor_list(num_neighbors_1: int, num_hops: int) -> List[int]:
        if num_hops > 2:
            return [num_neighbors_1, num_neighbors_1 // 2, num_neighbors_1 // 4]
        elif num_hops > 1:
            return [num_neighbors_1, num_neighbors_1 // 2]
        else:
            return [num_neighbors_1]

    return {
        'id': 'NeighborLoader',
        'num_neighbors': neighbor_list(num_neighbors_hop_1, num_hops),
        'replace': True,
        'batch_size': batch_size,
        'num_workers': 1
    }


def objective(trial) -> float:
    _model, class_weights = flickr_model(dataset_name='Flickr',
                                         hidden_channels=512,
                                         pooling_ratio=-1,
                                         dropout_p=0.2)
    network = training_env(trial, _model, class_weights)

    sampling_attributes = init_parameters_optimizer(trial)
    num_neighbors_str = '_'.join([str(count_neighbor) for count_neighbor in sampling_attributes['num_neighbors']])
    batch_size = sampling_attributes['batch_size']
    output_id = f'Flickr_n_{num_neighbors_str}_b_{batch_size}'
    train_loader, val_loader = loaders(sampling_attributes)

    network.train(output_id, _model, train_loader, val_loader)
    accuracy_history = network.performance_metrics.performance_values.get([MetricType.Accuracy], [-1.0])
    logging.info('Accuracy')
    return accuracy_history[-1]


if __name__ == "__main__":
    study = optuna.create_study(study_name='Flickr', direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


