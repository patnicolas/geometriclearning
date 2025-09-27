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
from typing import AnyStr, List, Any, Dict
import logging
from dataclasses import dataclass
# 3rd Party imports
from torch_geometric.nn import SAGEConv, GraphConv
import torch.nn as nn
# Library imports
from play import Play
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.training import TrainingException
from deeplearning.model.graph.graph_sage_model import GraphSAGEBuilder
from deeplearning.model.graph.graph_conv_model import GraphConvBuilder
import python


@dataclass(frozen=True)
class GraphSAGEvsGCNConfig:
    model_id: AnyStr
    num_layers: int
    neighbors: List[int]
    hidden_channels: int


class GraphSAGEvsGCNPlay(Play):
    """
    Source code related to the Substack article 'Graph Convolutional or GraphSAGE: shootout'. As with similar
        tutorial classes, model, training and neighborhood sampling are defined in declarative form (JSON string).
     Reference: https://patricknicolas.substack.com/p/graph-convolutional-or-sage-networks

    For sake of clarity, the traditional hyperparameters are fixed and only the parameters relevant to the comparison
    of the 2 models are considered:
    - Number of neighbors and fanout for message aggregation
    - Size of graph data
    - Number of graph neural layers.

    The features are implemented by the class GraphSAGEvsGCN in the source file
                  python/deeplearning/model/graph/graph_sage_vs_gcn.py
    The class GraphSAGEvsGCNPlay is a wrapper of the class GraphSAGEvsGCN
    The execution of the tests follows the same order as in the Substack article
    """
    # Hyperparameters fixed for evaluation
    lr: float = 0.0008
    epochs: int = 40
    dropout = 0.4

    def __init__(self, dataset_name: AnyStr, model_configs: List[GraphSAGEvsGCNConfig]) -> None:
        """
        Constructor for the comparison of GraphSAGE and GCN
        @param dataset_name: Name of PyTorch Geometric graph dataset
        @type dataset_name: str
        @param model_configs: List of model configuration to compare
        @type model_configs: List of GraphSAGEvsGCNConfig
        """
        super(GraphSAGEvsGCNPlay, self).__init__()

        self.model_configs = model_configs
        pyg_dataset = PyGDatasets(dataset_name)
        self.dataset = pyg_dataset()
        self.dataset_name = pyg_dataset.name

    def play(self) -> None:
        """
            Method to train, validate and compare several variant of GraphSAGE and GCN models
        """
        # Walk through the various models
        for model_config in self.model_configs:
            # Step 1: Load training parameters
            train_attrs = self.__get_training_attributes(model_config)
            # Step 2: Load the model parameters
            match model_config.model_id:
                case 'Conv':
                    model_attrs = self.__build_graph_conv(model_config)
                case 'SAGE':
                    model_attrs = self.__build_graph_sage(model_config)
                case _:
                    raise TrainingException(f'GNN {model_config.model_id} is not supported')

            logging.info(f'\nTraining attributes\n{train_attrs}\nModel attributes:\n{model_attrs}')
            # step 3: Load the neighborhood sampling attributes
            sampling_attrs = GraphSAGEvsGCNPlay.__get_sampling_attrs(model_config)
            # step 4: Execute training
            GraphSAGEvsGCNPlay.__execute_training(model_id=model_config.model_id,
                                                  training_attributes=train_attrs,
                                                  model_attributes=model_attrs,
                                                  sampling_attrs=sampling_attrs)

    """ ----------------------  Private Supporting Methods ------------------------  """

    def __get_training_attributes(self, tutorial_config: GraphSAGEvsGCNConfig):
        from dataset.graph.graph_data_loader import GraphDataLoader

        _data = self.dataset[0]
        class_weights = GraphDataLoader.class_weights(_data)

        # Parameterization
        return {
            'dataset_name': self.dataset_name,
            # Model training Hyperparameters
            'learning_rate': GraphSAGEvsGCNPlay.lr,
            'batch_size': 32,
            'loss_function': nn.CrossEntropyLoss(label_smoothing=0.05),
            'momentum': 0.95,
            'weight_decay': 2e-3,
            'encoding_len': -1,
            'train_eval_ratio': 0.9,
            'weight_initialization': 'Kaiming',
            'optim_label': 'adam',
            'drop_out': GraphSAGEvsGCNPlay.dropout,
            'is_class_imbalance': True,
            'class_weights': class_weights,
            'patience': 2,
            'min_diff_loss': 0.02,
            'epochs': GraphSAGEvsGCNPlay.epochs,
            # Model configuration
            'hidden_channels': tutorial_config.hidden_channels,
            # Performance metric definition
            'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR']
        }

    def __build_graph_conv(self,  tutorial_config: GraphSAGEvsGCNConfig) -> Dict[AnyStr, Any]:
        _data = self.dataset[0]
        title = self.__set_title(tutorial_config)

        return {
            'model_id': title,
            'graph_conv_blocks': [
                {
                    'block_id': 'MyBlock_1',
                    'conv_layer': GraphConv(in_channels=_data.num_node_features,
                                            out_channels=tutorial_config.hidden_channels),
                    'num_channels': tutorial_config.hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'pooling': None,
                    'dropout': GraphSAGEvsGCNPlay.dropout
                },
                {
                    'block_id': 'MyBlock_2',
                    'conv_layer': GraphConv(in_channels=tutorial_config.hidden_channels,
                                            out_channels=tutorial_config.hidden_channels),
                    'num_channels': tutorial_config.hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'pooling': None,
                    'dropout': GraphSAGEvsGCNPlay.dropout
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'Output',
                    'in_features': tutorial_config.hidden_channels,
                    'out_features': self.dataset.num_classes,
                    'activation': None
                }
            ]
        }

    def __build_graph_sage(self, tutorial_config: GraphSAGEvsGCNConfig) -> Dict[AnyStr, Any]:
        _data = self.dataset[0]
        title = self.__set_title(tutorial_config)

        return {
            'model_id': f'Graph{title}',
            'graph_SAGE_blocks': [
                {
                    'block_id': 'SAGE Layer 1',
                    'SAGE_layer': SAGEConv(in_channels=self.dataset[0].num_node_features,
                                           out_channels=tutorial_config.hidden_channels),
                    'num_channels': tutorial_config.hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': GraphSAGEvsGCNPlay.dropout
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=tutorial_config.hidden_channels,
                                           out_channels=tutorial_config.hidden_channels),
                    'num_channels': tutorial_config.hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': GraphSAGEvsGCNPlay.dropout
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'Node classification block',
                    'in_features': tutorial_config.hidden_channels,
                    'out_features': self.dataset.num_classes,
                    'activation': None
                }
            ]
        }

    def __set_title(self, tutorial_config: GraphSAGEvsGCNConfig) -> AnyStr:
        return (f'{tutorial_config.model_id}_{self.dataset_name}_Neighbors{tutorial_config.neighbors}_'
                f'{tutorial_config.num_layers}layers')

    @staticmethod
    def __get_sampling_attrs(tutorial_config: GraphSAGEvsGCNConfig) -> Dict[AnyStr, Any]:
        return {
            'id': 'NeighborLoader',
            'num_neighbors': tutorial_config.neighbors,
            'batch_size': 32,
            'replace': True,
            'num_workers': 4
        }

    @staticmethod
    def __execute_training(model_id: AnyStr,
                           training_attributes: Dict[AnyStr, Any],
                           model_attributes: Dict[AnyStr, Any],
                           sampling_attrs: Dict[AnyStr, Any],
                           num_subgraph_nodes: int = 0) -> None:
        from deeplearning.training.gnn_training import GNNTraining
        from dataset.graph.graph_data_loader import GraphDataLoader

        # Step 1:
        graph_builder = GraphSAGEBuilder(model_attributes) if model_id == 'SAGE' else GraphConvBuilder(model_attributes)
        graph_model = graph_builder.build()

        # Step 2:  Create the trainer using the training attributes dictionary
        trainer = GNNTraining.build(training_attributes)
        # Step 3: Create the data loader and extract a sub graph
        graph_data_loader = GraphDataLoader(dataset_name=training_attributes['dataset_name'],
                                            sampling_attributes=sampling_attrs,
                                            num_subgraph_nodes=num_subgraph_nodes)
        logging.info(graph_data_loader)
        train_loader, eval_loader = graph_data_loader()
        # Step 4: Train and Validate the model
        graph_model.train_model(trainer, train_loader, eval_loader)


if __name__ == '__main__':
    try:
        model1 = GraphSAGEvsGCNConfig(model_id='Conv', num_layers=2, neighbors=[6, 3], hidden_channels=64)
        model2 = GraphSAGEvsGCNConfig(model_id='Conv', num_layers=4, neighbors=[6, 3], hidden_channels=64)
        model3 = GraphSAGEvsGCNConfig(model_id='SAGE', num_layers=2, neighbors=[6, 3], hidden_channels=64)
        model4 = GraphSAGEvsGCNConfig(model_id='SAGE', num_layers=4, neighbors=[6, 3], hidden_channels=64)
        tutorial = GraphSAGEvsGCNPlay(dataset_name='Cora', model_configs=[model1, model2, model3, model4])
        tutorial.play()
    except AssertionError as e:
        logging.error(e)
        assert False
    except TrainingException as e:
        logging.error(e)
        assert False


