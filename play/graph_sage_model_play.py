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
from typing import AnyStr, Any, Dict
import logging
# 3rd Party imports
from torch_geometric.nn import SAGEConv
import torch.nn as nn
# Library imports
from play import Play
from dataset.graph.graph_data_loader import GraphDataLoader
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.training import TrainingException
from deeplearning.model.graph.graph_sage_model import GraphSAGEBuilder
from deeplearning.training.gnn_training import GNNTraining
import python


class GraphSAGEModelPlay(Play):
    """
    Source code related to the Substack article 'Revisiting Inductive Graph Neural Networks'. As with similar
    tutorial classes, model, training and neighborhood sampling are defined in declarative form (JSON string).

    Article: https://patricknicolas.substack.com/p/revisiting-inductive-graph-neural
    GraphSAGE model:
        https://github.com/patnicolas/geometriclearning/blob/main/python/deeplearning/model/graph/graph_sage_model.py

    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self,
                 dataset_name: AnyStr,
                 model_attributes: Dict[AnyStr, Any],
                 training_attributes: Dict[AnyStr, Any],
                 sampling_attributes: Dict[AnyStr, Any]) -> None:
        super(GraphSAGEModelPlay, self).__init__()

        self.dataset_name = dataset_name
        self.model_attributes = model_attributes
        self.training_attributes = training_attributes
        self.sampling_attributes = sampling_attributes

    def play(self) -> None:
        """
        Implementation of the evaluation of GraphSAGE model as described in Substack article ''Revisiting Inductive
        Graph Neural Networks' - Code snippet 7
        """
        # Step 1:
        graph_SAGE_builder = GraphSAGEBuilder(self.model_attributes)
        graph_SAVE_model = graph_SAGE_builder.build()
        # Step 2:  Create the trainer using the training attributes dictionary
        trainer = GNNTraining.build(self.training_attributes)
        # Step 3: Create the data loader and extract a sub graph
        graph_data_loader = GraphDataLoader(dataset_name=self.dataset_name,
                                            sampling_attributes=self.sampling_attributes,
                                            num_subgraph_nodes=-1)
        logging.info(graph_data_loader)
        train_loader, eval_loader = graph_data_loader()
        # Step 4: Train and Validate the model
        graph_SAVE_model.train_model(trainer, train_loader, eval_loader)


if __name__ == '__main__':
    test_dataset_name = 'Cora'
    lr = 0.001
    pyg_dataset = PyGDatasets('PubMed')
    dataset = pyg_dataset()
    epochs = 80
    hidden_channels = 128
    class_weights = GraphDataLoader.class_weights(dataset[0])
    neighbors = [6, 3]
    num_layers = 2
    title = f'SAGE_{test_dataset_name}_NeighborLoader{neighbors}_{num_layers}layers'

    # Parameterization of the training attributes
    test_training_attributes = {
        'dataset_name': test_dataset_name,
        # Model training Hyperparameters
        'learning_rate': lr,
        'batch_size': 32,
        'loss_function': nn.CrossEntropyLoss(label_smoothing=0.05),
        'momentum': 0.95,
        'weight_decay': 2e-3,
        'encoding_len': -1,
        'train_eval_ratio': 0.9,
        'weight_initialization': 'Kaiming',
        'optim_label': 'adam',
        'drop_out': 0.25,
        'is_class_imbalance': True,
        'class_weights': class_weights,
        'patience': 2,
        'min_diff_loss': 0.02,
        'epochs': epochs,
        # Model configuration
        'hidden_channels': hidden_channels,
        # Performance metric definition
        'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR']
    }

    # Parameterized model attributes
    test_model_attributes = {
        'model_id': f'Graph{title}',
        'graph_SAGE_blocks': [
            {
                'block_id': 'SAGE Layer 1',
                'SAGE_layer': SAGEConv(in_channels=dataset[0].num_node_features, out_channels=hidden_channels),
                'num_channels': hidden_channels,
                'activation': nn.ReLU(),
                'batch_norm': None,
                'dropout': 0.4
            },
            {
                'block_id': 'SAGE Layer 2',
                'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                'num_channels': hidden_channels,
                'activation': nn.ReLU(),
                'batch_norm': None,
                'dropout': 0.4
            },
            {
                'block_id': 'SAGE Layer 2',
                'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                'num_channels': hidden_channels,
                'activation': nn.ReLU(),
                'batch_norm': None,
                'dropout': 0.4
            },
            {
                'block_id': 'SAGE Layer 2',
                'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                'num_channels': hidden_channels,
                'activation': nn.ReLU(),
                'batch_norm': None,
                'dropout': 0.4
            }
        ],
        'mlp_blocks': [
            {
                'block_id': 'Node classification block',
                'in_features': hidden_channels,
                'out_features': dataset.num_classes,
                'activation': None
            }
        ]
    }

    #  Parameterized of neighborhood sampling for message aggregation
    test_sampling_attributes = {
        'id': 'NeighborLoader',
        'num_neighbors': neighbors,
        'batch_size': 32,
        'replace': True,
        'num_workers': 4
    }

    try:
        graph_sage_model_tutorial = GraphSAGEModelPlay(dataset_name=test_dataset_name,
                                                       model_attributes=test_model_attributes,
                                                       training_attributes=test_training_attributes,
                                                       sampling_attributes=test_sampling_attributes)
        graph_sage_model_tutorial.play()
        assert True
    except AssertionError as e:
        logging.error(e)
        assert False
    except TrainingException as e:
        logging.error(e)
        assert False



