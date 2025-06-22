__author__ = "Patrick Nicolas"
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

from dl import GNNException
import torch.nn as nn
import torch_geometric
from dataset.graph.graph_data_loader import GraphDataLoader
from dataset.graph.pyg_datasets import PyGDatasets
from torch_geometric.nn import GraphConv
from dl.model.gconv_model import GConvModel
import torch
from torch_geometric.data import Data
from typing import AnyStr, Dict, Any
from dl.training.gnn_training import GNNTraining
from torch.utils.data import DataLoader


class EvalGConv(object):
    def __init__(self,
                 _training_attributes: Dict[AnyStr, Any],
                 _sampling_attributes: Dict[AnyStr, Any]) -> None:
        self.training_attributes = _training_attributes
        self.sampling_attributes = _sampling_attributes

    def start_training(self) -> None:
        # Step 1: Retrieve the evaluation model
        flickr_model, class_weights = self.__get_eval_model()
        num_neighbors_str = '-'.join([str(n) for n in self.sampling_attributes['num_neighbors']])
        title = f"{self.training_attributes['dataset_name']}_neighbors_sampling_{num_neighbors_str}"

        # Step 2: Retrieve the training environment
        gnn_training = self.__get_training_env(flickr_model, class_weights)

        # Step 3: Retrieve the training and validation data loader
        train_loader, val_loader = self.__get_loaders()

        # Step 4: Train the model
        gnn_training.train(model_id=title,
                           neural_model=flickr_model,
                           train_loader=train_loader,
                           val_loader=val_loader)

    """ --------------------------  Private Helper Methods -----------------------  """

    def __get_loaders(self) -> (DataLoader, DataLoader):
        flickr_loaders = GraphDataLoader(dataset_name=self.training_attributes['dataset_name'],
                                         sampling_attributes=self.sampling_attributes)
        logging.info(f'Graph data: {str(flickr_loaders.data)}')
        train_loader, val_loader = flickr_loaders()
        return train_loader, val_loader

    @staticmethod
    def __distribution(data: Data) -> torch.Tensor:
        class_distribution = data.y[data.train_mask]
        raw_distribution = torch.bincount(class_distribution)
        raw_weights = 1.0/raw_distribution
        return raw_weights/raw_weights.sum()

    def __get_training_env(self, model: GConvModel, class_weights: torch.Tensor = None) -> GNNTraining:
        self.training_attributes['loss_function'] = nn.NLLLoss(weight=class_weights.to('mps')) \
            if class_weights is not None \
            else nn.NLLLoss()
        self.training_attributes['encoding_len'] = model.mlp_blocks[-1].get_out_features()
        self.training_attributes['class_weights'] = class_weights
        return GNNTraining.build(self.training_attributes)

    def __get_eval_model(self) -> (GConvModel, torch.Tensor):
        from torch_geometric.datasets.flickr import Flickr

        pyg_dataset = PyGDatasets(self.training_attributes['dataset_name'])
        flickr_dataset: Flickr = pyg_dataset()
        if flickr_dataset is None:
            raise GNNException("Failed to load Flickr")

        _data: torch_geometric.data.Data = flickr_dataset[0]
        logging.info(f'Number of features: {_data.num_node_features}\nNumber of classes: {flickr_dataset.num_classes}'
              f'\nSize of training: {_data.train_mask.sum()}')

        my_model = self.__get_model(num_node_features=_data.num_node_features,
                                    _num_classes=flickr_dataset.num_classes,
                                    hidden_channels=384)
        return my_model, EvalGConv.__distribution(_data)

    def __get_model(self, num_node_features: int, _num_classes: int, hidden_channels: int) -> GConvModel:
        model_attributes = {
            'model_id': 'MyModel',
            'gconv_blocks': [
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
                    'out_features': _num_classes,
                    'activation': nn.LogSoftmax(dim=-1),
                    'dropout': 0.0
                }
            ]
        }
        return GConvModel.build(model_attributes)


if __name__ == '__main__':
    training_attributes = {
        'dataset_name': 'Flickr',
        # Model training Hyperparameters
        'learning_rate': 0.0005,
        'momentum': 0.90,
        'batch_size': 64,
        'loss_function': None,
        'encoding_len': -1,
        'train_eval_ratio': 0.9,
        'epochs': 24,
        'weight_initialization': 'xavier',
        'optim_label': 'adam',
        'drop_out': 0.25,
        'is_class_imbalance': True,
        'class_weights': None,
        'patience': 2,
        'min_diff_loss': 0.02,
        # Performance metric definition
        'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'plot_parameters': [
            {'title': 'Accuracy', 'x_label': 'epoch', 'y_label': 'accuracy'},
            {'title': 'Precision', 'x_label': 'epochs', 'y_label': 'precision'},
            {'title': 'Recall', 'x_label': 'epochs', 'y_label': 'recall'},
            {'title': 'F1', 'x_label': 'epochs', 'y_label': 'F1'},
        ]
    }
    sampling_attributes = {
        'id': 'NeighborLoader',
        'num_neighbors': [4],
        'batch_size': 64,
        'replace': True,
        'num_workers': 4
    }

    eval_gconv = EvalGConv(training_attributes, sampling_attributes)
    eval_gconv.start_training()


