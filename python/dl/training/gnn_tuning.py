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
from torch_geometric.data import Data, Dataset

from dl.training.graph_hyperparams_tuning import distribution
from metric.metric_type import MetricType
from typing import Dict, List, AnyStr, Any
from dl.training.gnn_training import GNNTraining
from torch.utils.data import DataLoader


"""
Wrapper for tuning a Graph Convolutional Neural Network. The optuna library of te 
"""
class GNNTuning(object):
    gnn_conv_model: GConvModel = None
    training_parameters: Dict[AnyStr, Any] = None

    @staticmethod
    def loaders(sampling_attributes: Dict[AnyStr, Any]) -> (DataLoader, DataLoader):
        """
        Abstract the loading of the data set given a set of sampling attributes to be searched
        @param sampling_attributes: Sampling attributes that need to be searched as part of the optimization of
        the sampling
        @type sampling_attributes: Dict[AnyStr, Any]
        :return: Pair of training and evaluation data loaders
        :rtype: Tuple[DataLoader, DataLoader]
        """
        flickr_loaders = GraphDataLoader(dataset_name='Flickr', sampling_attributes=sampling_attributes)
        train_loader, val_loader = flickr_loaders()
        return train_loader, val_loader

    @staticmethod
    def init_parameters_optimizer(trial) -> Dict[AnyStr, Any]:
        """
        Initialization of the graph network sampling parameters to be optimized
        @param trial:
        @type trial:
        @return:
        @rtype:
        """
        # These are the 2 sampling parameters we need to optimize
        num_neighbors_hop_1 = trial.suggest_categorical('num_neighbors_1', [4, 8, 12, 24])
        num_hops = trial.suggest_categorical('num_hops', [2, 3])

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
            'batch_size': 128,
            'num_workers': 1
        }

    @staticmethod
    def objective(trial) -> float:
        assert GNNTuning.gnn_conv_model is not None, 'Graph Convolutional model is not defined'
        assert GNNTuning.training_parameters is not None, 'Training attributes are not defined'

        gnn_training = GNNTraining.build(GNNTuning.training_parameters)
        sampling_attributes = GNNTuning.init_parameters_optimizer(trial)
        num_neighbors_str = '_'.join([str(count_neighbor) for count_neighbor in sampling_attributes['num_neighbors']])
        batch_size = sampling_attributes['batch_size']
        output_id = f'Flickr_n_{num_neighbors_str}_b_{batch_size}'
        train_loader, val_loader = GNNTuning.loaders(sampling_attributes)

        gnn_training.train(output_id, GNNTuning.gnn_conv_model, train_loader, val_loader)
        accuracy_history = gnn_training.performance_metrics.performance_values.get([MetricType.Accuracy], [-1.0])
        return accuracy_history[-1]

    @staticmethod
    def __load_dataset(training_parameters: Dict[AnyStr, Any]) -> Dataset:
        from torch_geometric.datasets.flickr import Flickr
        target_device = 'mps'
        dataset_name = training_parameters['dataset_name']
        pyg_dataset = PyGDatasets(dataset_name)
        _dataset: Flickr = pyg_dataset()
        if _dataset is None:
            raise GNNException("Failed to load Flickr")

        _data: torch_geometric.data.Data = _dataset[0]
        if training_parameters['is_class_imbalance']:
            training_parameters['class_weights'] = distribution(_data)
            training_parameters['loss_function'] = nn.NLLLoss(weight=training_parameters['class_weights'].to(target_device))
        else:
            training_parameters['loss_function'] = nn.NLLLoss()
        print(f'Number of features: {_data.num_node_features}\nNumber of classes: {_dataset.num_classes}'
              f'\nSize of training: {_data.train_mask.sum()}')
        return _dataset

    @staticmethod
    def __initialize_training_parameters() -> Dict[AnyStr, Any]:
        return {
            'dataset_name': 'Flickr',
            # Model training Hyperparameters
            'learning_rate': 0.0005,
            'batch_size': 64,
            'loss_function': None,
            'momentum': 0.90,
            'encoding_len': -1,
            'train_eval_ratio': 0.9,
            'weight_initialization': 'xavier',
            'optim_label': 'adam',
            'drop_out': 0.25,
            'is_class_imbalance': True,
            'class_weights': None,
            'patience': 2,
            'min_diff_loss': 0.02,
            # Model configuration
            'hidden_channels': 256,
            # Performance metric definition
            'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1'],
            'plot_parameters': [
                {'title': 'Accuracy', 'x_label': 'epoch', 'y_label': 'accuracy'},
                {'title': 'Precision', 'x_label': 'epochs', 'y_label': 'precision'},
                # ....
            ]
        }

    @staticmethod
    def __get_model(training_parameters: Dict[AnyStr, Any], _dataset: Dataset) -> GConvModel:
        _data = _dataset[0]

        # First graph convolutional layer and neural block
        conv_1 = GraphConv(in_channels=_data.num_node_features, out_channels=training_parameters['hidden_channels'])
        gconv_block_1 = GConvBlock(block_id='Conv 24-256',
                                   gconv_layer=conv_1,
                                   batch_norm_module=None,
                                   activation_module=nn.ReLU(),
                                   dropout_module=nn.Dropout(training_parameters['drop_out']))

        # Second graph convolutional layer and neural block
        conv_2 = GraphConv(in_channels=training_parameters['hidden_channels'],
                           out_channels=training_parameters['hidden_channels'])
        gconv_block_2 = GConvBlock(block_id='Conv 256-256',
                                   gconv_layer=conv_2,
                                   batch_norm_module=None,
                                   activation_module=nn.ReLU(),
                                   dropout_module=nn.Dropout(training_parameters['drop_out']))

        # Fully connected output layer for classification
        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(training_parameters['hidden_channels'], _dataset.num_classes),
                             activation_module=nn.LogSoftmax(dim=-1))
        return GConvModel(model_id='Flicker test dataset',
                          gconv_blocks=[gconv_block_1, gconv_block_2],
                          mlp_blocks=[mlp_block])

    @staticmethod
    def eval_model() -> None:
        GNNTuning.training_parameters = GNNTuning.__initialize_training_parameters()
        flickr_dataset = GNNTuning.__load_dataset(GNNTuning.training_parameters)
        GNNTuning.gnn_conv_model = GNNTuning.__get_model(GNNTuning.training_parameters, flickr_dataset)


if __name__ == "__main__":
    GNNTuning.eval_model()
    study = optuna.create_study(study_name='Flickr', direction="maximize")
    study.optimize(GNNTuning.objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

