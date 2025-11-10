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


# Python standard library imports
from typing import AnyStr, Dict, Any
import logging
# 3rd Party imports
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphConv

from deeplearning.training.exec_config import ExecConfig
# Library imports
from play import Play
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.training.gnn_training import GNNTraining
from deeplearning.model.graph.graph_conv_model import GraphConvModel, GraphConvBuilder
from dataset import DatasetException
from dataset.graph.graph_data_loader import GraphDataLoader
from deeplearning.block.graph import GraphException
import python

__all__ = ['GNNTrainingPlay']


class GNNTrainingPlay(Play):
    """
    Source code related to the Substack article 'Plug & Play Training for Graph Convolutional Networks'.
    As with similar tutorial classes, model, training and neighborhood sampling are defined in declarative form
    (JSON string).

    Reference: https://patricknicolas.substack.com/p/plug-and-play-training-for-graph
    GraphSAGE model:
        https://github.com/patnicolas/geometriclearning/blob/main/python/deeplearning/model/graph/graph_sage_model.py

    The features are implemented by the class GNNTraining in the source file
                  python/deeplearning/training/gnn_training.py
    The class GNNTrainingPlay is a wrapper of the class GNNTraining
    """
    def __init__(self,
                 training_attributes: Dict[AnyStr, Any],
                 sampling_attributes: Dict[AnyStr, Any],
                 model_attributes: Dict[AnyStr, Any] = None) -> None:
        super(GNNTrainingPlay, self).__init__()
        assert len(training_attributes) > 0, 'Training attributes are undefined'
        assert len(sampling_attributes) > 0, 'Sampling attributes are undefined'

        self.training_attributes = training_attributes
        self.sampling_attributes = sampling_attributes
        self.model_attributes = None

    def play(self) -> None:
        """
        Source code related to Substack article 'Plug & Play Training for Graph Convolutional Networks' -
        Code snippets, 6, 7, 8, 9, 10 & 11
        Ref:  https://patricknicolas.substack.com/p/plug-and-play-training-for-graph
        """
        # Step 1: Retrieve the evaluation model
        this_model, class_weights = self.get_eval_model()

        # Step 2: Retrieve the training environment
        gnn_training = self.get_training_env(this_model, class_weights)

        # Step 3: Retrieve the training and validation data loader
        train_loader, val_loader = self.get_loaders()

        # Step 4: Train the model using the GNNTraining class
        gnn_training.train(neural_model=this_model,
                           train_loader=train_loader,
                           val_loader=val_loader)

    def get_loaders(self) -> (DataLoader, DataLoader):
        graph_data_loaders = GraphDataLoader(dataset_name=self.training_attributes['dataset_name'],
                                             sampling_attributes=self.sampling_attributes)
        logging.info(f'Graph data: {str(graph_data_loaders.data)}')
        train_loader, val_loader = graph_data_loaders()
        return train_loader, val_loader

    def get_training_env(self, model: GraphConvModel, class_weights: torch.Tensor = None) -> GNNTraining:
        target_device = self.training_attributes.get('target_device', 'cpu')
        if class_weights is not None and target_device is not None:
            class_weights = class_weights.to(target_device)
            self.training_attributes['loss_function'] = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights)
        else:
            self.training_attributes['loss_function'] = nn.CrossEntropyLoss(label_smoothing=0.05)

        self.training_attributes['encoding_len'] = model.mlp_blocks[-1].get_out_features()
        self.training_attributes['class_weights'] = class_weights
        exec_config = ExecConfig(empty_cache=True,
                                 mix_precision=False,
                                 subset_size=0,
                                 monitor_memory=True,
                                 grad_accu_steps=1,
                                 pin_mem=True)
        return GNNTraining.build(self.training_attributes, exec_config)

    def get_eval_model(self) -> (GraphConvModel, torch.Tensor):
        pyg_datasets = PyGDatasets(self.training_attributes['dataset_name'])
        this_dataset = pyg_datasets()
        if this_dataset is None:
            raise GraphException(f"Failed to load {self.training_attributes['dataset_name']}")

        _data: Data = this_dataset[0]
        logging.info(f'Number of features: {_data.num_node_features}\nNumber of classes: {this_dataset.num_classes}'
                     f'\nSize of training: {_data.train_mask.sum()}')

        # Use the model attributes provided in the constructor if not None, otherwise load a default model.
        my_model = GNNTrainingPlay.__get_model(num_node_features=_data.num_node_features,
                                               num_classes=this_dataset.num_classes,
                                               hidden_channels=self.training_attributes['hidden_channels']) \
            if self.model_attributes is None  \
            else self.model_attributes
        return my_model, GNNTrainingPlay.__distribution(_data)

    """ --------------------------  Private Helper Methods -----------------------  """

    @staticmethod
    def __get_model(num_node_features: int, num_classes: int, hidden_channels: int) -> GraphConvModel:
        model_attributes = {
            'model_id': 'MyModel',
            'graph_conv_blocks': [
                {
                    'block_id': 'MyBlock_1',
                    'conv_layer': GraphConv(in_channels=num_node_features, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'pooling': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'MyBlock_2',
                    'conv_layer': GraphConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'pooling': None,
                    'dropout': 0.25
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'MyMLP',
                    'in_features': hidden_channels,
                    'out_features': num_classes,
                    'activation': None,
                    'dropout': 0.0
                }
            ]
        }
        return GraphConvBuilder(model_attributes).build()

    @staticmethod
    def __distribution(data: Data) -> torch.Tensor:
        class_distribution = data.y[data.train_mask]
        raw_distribution = torch.bincount(class_distribution)
        raw_weights = 1.0 / raw_distribution
        return raw_weights / raw_weights.sum()


if __name__ == '__main__':
    default_title = 'Flickr Neighbors Sampling'
    training_attrs = {
        'target_device': None,   # To be selected dynamically
        'dataset_name': 'Flickr',
        # Model training Hyperparameters
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'momentum': 0.90,
        'batch_size': 64,
        'loss_function': None,
        'encoding_len': -1,
        'train_eval_ratio': 0.9,
        'epochs': 3,
        'weight_initialization': 'Kaiming',
        'optim_label': 'adam',
        'drop_out': 0.25,
        'is_class_imbalance': True,
        'class_weights': None,
        'patience': 2,
        'min_diff_loss': 0.02,
        'hidden_channels': 128,
        'tensor_mix_precision': None,
        'checkpoint_enabled': False,
        # Performance metric definition
        'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR'],
        'plot_parameters': {
            'count': 0,
            'x_label': 'Epochs',
            'x_label_size': 12,
            'title': '',   # To be updated dynamically
            'fig_size': (10, 10),
            'plot_filename': ''  # To be updated dynamically
        }
    }
    sampling_attrs = {
        'id': 'NeighborLoader',
        'num_neighbors': [4],
        'batch_size': 64,
        'replace': True,
        'pin_memory': False,
        'num_workers': 4
    }

    try:
        gnn_training_play = GNNTrainingPlay(training_attrs, sampling_attrs)
        gnn_training_play.play()
        assert True
    except (GraphException, DatasetException, AssertionError, ValueError) as e:
        logging.error(f'Error: {str(e)}')
        assert False
