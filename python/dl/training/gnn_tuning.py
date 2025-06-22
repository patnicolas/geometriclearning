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

import optuna
from optuna.trial import TrialState

import torch.nn as nn
from dataset.graph.graph_data_loader import GraphDataLoader
from torch_geometric.nn import GraphConv
from dl.model.gconv_model import GConvModel
from torch_geometric.data import Data
from metric.metric_type import MetricType
from typing import Dict, List, AnyStr, Any, Callable
from dl.training.gnn_training import GNNTraining
import torch


def class_weight_distribution(data: Data) -> List[int]:
    """
    Standard method to compute the weight distribution of classes or labels in case of the class imbalance
    @param data: Graph data describing node and edge indices
    @type data: torch_geometric.data.Dta
    @return: Weights for each class as a 1 dimension Torch tensor
    @rtype: torch.Tensor
    """
    class_distribution = data.y[data.train_mask]
    raw_distribution = torch.bincount(class_distribution)
    raw_weights = 1.0 / raw_distribution
    return raw_weights / raw_weights.sum()


def neighbor_list_generator(num_neighbors_1: int, num_hops: int) -> List[int]:
    """
    Function that generates a list of number of neighbors to be sampled for each node per hops
    @param num_neighbors_1: Number of sampled neighboring nodes in the first hop
    @type num_neighbors_1: int
    @param num_hops: Number of hops
    @type num_hops: int
    @return: List of neighboring sampled nodes for each hop
    @rtype: List[int]
    """
    if num_hops > 2:
        return [num_neighbors_1, num_neighbors_1 // 2, num_neighbors_1 // 4]
    elif num_hops > 1:
        return [num_neighbors_1, num_neighbors_1 // 2]
    else:
        return [num_neighbors_1]


"""
Wrapper for tuning a Graph Convolutional Neural Network using the Optima HPO. 
Optimizing the many parameters involved in training and evaluating a Graph Neural Network can be daunting—even 
for experienced practitioners. To simplify this process, we divide the search for the optimal 
architecture and training configuration into three distinct steps,
This implementation uses accuracy as target metric.
"""

class GNNTuning(object):
    objective_metric = MetricType.Accuracy
    target_device = 'mps'
    # Dynamic (mutable) training parameters
    training_parameters: Dict[AnyStr, Any] = {
        'dataset_name': 'Flickr',
        'learning_rate': 0.0005,
        'batch_size': 64,
        'loss_function': None,   # To be initialize
        'momentum': 0.90,
        'encoding_len': -1,
        'train_eval_ratio': 0.9,
        'weight_initialization': 'xavier',
        'epochs': 32,
        'optim_label': 'adam',
        'drop_out': 0.25,
        'is_class_imbalance': True,
        'class_weights': None,   # To be updated if necessary
        'patience': 2,
        'min_diff_loss': 0.02,
        'hidden_channels': 256,
        'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'plot_parameters': [
            {'title': 'Accuracy', 'x_label': 'epochs', 'y_label': 'Accuracy'},
            {'title': 'Precision', 'x_label': 'epochs', 'y_label': 'Precision'},
            {'title': 'Recall', 'x_label': 'epochs', 'y_label': 'Recall'},
            {'title': 'F1', 'x_label': 'epochs', 'y_label': 'F1'},
        ]
    }

    @staticmethod
    def objective(trial) -> float:
        """
        Objective function for a given Optuna trial (hyperparameter configuration). The 7 steps are documented
        in the code. This implementation uses accuracy but other metric such as precision, recall,.. can be
        used as well as validation loss.
        Note: validation loss will require to minimize as direction of the study
        @param trial: Optuna trail
        @return: Objective value (i.e. accuracy of the last epoch for a given trial)
        @rtype: float
        """
        # Step 1: Initialize the Hyperparameters for the neighbor graph node sampling method
        sampling_attributes = GNNTuning.__init_hpo(trial, neighbor_list_generator)

        # Step 2: Extract loader for training and validation data
        graph_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=sampling_attributes)
        train_loader, val_loader = graph_loader()

        # Step 3: Initialize/update loss function as a hyperparameter
        GNNTuning.__init_loss_function(graph_loader.data, class_weight_distribution)

        # Step 4: Set up the training environment
        gnn_training = GNNTraining.build(GNNTuning.training_parameters)

        # For debugging purpose
        logging.info(f'Number of features: {graph_loader.data.num_node_features}n'
              f'\nNumber of classes: {graph_loader.num_classes}'
              f'\nSize of training: {graph_loader.data.train_mask.sum()}')

        # Step 5: Initialize the model using the JSON descriptive format
        gnn_conv_model = GNNTuning.__get_model(data=graph_loader.data,
                                               num_classes=graph_loader.num_classes,
                                               hidden_channels=384)
        # Step 6: Train and validate the model
        gnn_training.train(model_id=GNNTuning.__get_output_id(sampling_attributes),
                           neural_model=gnn_conv_model,
                           train_loader=train_loader,
                           val_loader=val_loader)

        # Step 7: Select the metric (Accuracy, Precision,...) for objective function to maximize
        metric_history = gnn_training.get_metric_history(GNNTuning.objective_metric)
        # We return the last metric of the epoch as objective value
        return metric_history[-1]

    """ --------------------------  Private Helper Methods -----------------------  """

    @staticmethod
    def __init_hpo(trial,
                   neighbor_sampling_func: Callable[[int, int], List[int]]) -> Dict[AnyStr, Any]:
        """
        Initialization of the graph network sampling parameters using a user-defined function
        @param trial: Optuna trial (hyperparameter configuration)
        @param neighbor_sampling_func: Function to generate the list of sampled neighboring nodes for
                a given graph node
        @type neighbor_sampling_func: Callable (num_neighbors, hop) => list neighbors per hop
        @return: Finalized graph node sampling configuration to evaluate
        @rtype: Dictionary
        """
        # These are the 2 sampling parameters we need to optimize
        num_neighbors_hop_1 = trial.suggest_categorical('num_neighbors_1', [4, 8, 12, 24])
        num_hops = trial.suggest_categorical('num_hops', [2, 3])

        return {
            'id': 'NeighborLoader',
            'num_neighbors': neighbor_sampling_func(num_neighbors_hop_1, num_hops),
            'replace': True,
            'batch_size': 128,
            'num_workers': 1
        }

    @staticmethod
    def __init_loss_function(_data: Data, class_weight_func: Callable[[Data], List[int]]) -> None:
        """
        Potentially update the loss function with data and distribution of class frequency weight if
        the frequency of instance count per class is imbalanced
        @param _data: Graph data associated with this dataset
        @type _data: torch_geometric.util.data.Data
        @param class_weight_func: Function to generate the weight distribution for imbalance classes
        @type : Callable (data) => list of class weights
        """
        if GNNTuning.training_parameters['is_class_imbalance']:
            class_weights = class_weight_func(_data)
            GNNTuning.training_parameters['class_weights'] = class_weights
            GNNTuning.training_parameters['loss_function'] = (
                nn.NLLLoss(weight=class_weights.to(GNNTuning.target_device)))
        # Assign the default loss if classes are not imbalance
        else:
            GNNTuning.training_parameters['loss_function'] = nn.NLLLoss()

    @staticmethod
    def __get_model(data: Data, num_classes: int, hidden_channels: int) -> GConvModel:
        num_node_features = data.num_node_features
        model_parameters = {
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
                    'out_features': num_classes,
                    'activation': nn.LogSoftmax(dim=-1),
                    'dropout': 0.0
                }
            ]
        }
        return GConvModel.build(model_parameters)

    @staticmethod
    def __get_output_id(sampling_parameters: Dict[AnyStr, Any]) -> AnyStr:
        num_neighbors_str = '_'.join([str(count_neighbor) for count_neighbor in sampling_parameters['num_neighbors']])
        batch_size = sampling_parameters['batch_size']
        return f'Flickr_n_{num_neighbors_str}_b_{batch_size}'


if __name__ == "__main__":
    # We select the Tree-based Parzen Estimator for our HPO
    from optuna.samplers import TPESampler

    # We selected arbitrary Accuracy as our objective to maximize
    # The validation loss as our objective would require the direction 'minimize'
    study = optuna.create_study(study_name='Flickr', sampler=TPESampler(), direction="maximize")
    study.optimize(GNNTuning.objective, n_trials=100, timeout=600)

    # We select no deepcopy (=False) for memory efficient. There is no need
    # to preserve independent copies of each trial object.
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

