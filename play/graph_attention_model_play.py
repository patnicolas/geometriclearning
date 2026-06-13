__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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
import python
# 3rd party library imports
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset
# Library imports
from play import Play
from dataset import DatasetException
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.training.gnn_training import GNNTraining
from deeplearning.model.graph.graph_attention_model import GraphAttentionBuilder
from dataset.graph.graph_data_loader import GraphDataLoader
from deeplearning.block.graph import GraphException


class GraphAttentionModelPlay(Play):
    """
    Source code related to the Substack article 'Graphs Deserve Some Attention'.
    Reference: https://patricknicolas.substack.com/p/graphs-deserve-some-attention

    Source code for Graph Attention Network
    https://github.com/patnicolas/geometriclearning/blob/main/python/dataset/graph/graph_data_loader.py

    The features are implemented by the class GraphAttentionModel in the source file
                    python/deeplearning/graph/graph_attention_model.py
    The class GraphAttentionModelPlay is a wrapper of the class GraphAttentionModel
    """
    def __init__(self, dataset_name: AnyStr, neighbors: List[int], hidden_channels: int, num_heads: int) -> None:
        super(GraphAttentionModelPlay, self).__init__()
        self.dataset_name = dataset_name
        self.neighbors = neighbors
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

    def play(self) -> None:
        try:
            # 1.  Load the data set using PyTorch Geometry predefined data source
            pyg_dataset = PyGDatasets(self.dataset_name)
            dataset = pyg_dataset()

            # Initialize the 3 groups of attributes
            training_attributes = self.__get_training_attributes(dataset)
            loader_attributes = self.__get_loader_attributes()
            model_attributes = self.__get_model_attributes(
                title=f'Graph_Attention_{self.dataset_name}_Neighbors{self.neighbors}',
                dataset=dataset
            )

            # 3. Instantiate a graph attention model from the JSON attributes
            graph_attention_builder = GraphAttentionBuilder(model_attributes)
            graph_attention_model = graph_attention_builder.build()

            # 4. Load one of the PyTorch Geometric Dataset - training and evaluation
            graph_data_loader = GraphDataLoader(dataset_name=self.dataset_name, sampling_attributes=loader_attributes)
            train_loader, eval_loader = graph_data_loader()

            # 5. Setup and execute the training/evaluation of the model
            trainer = GNNTraining.build(training_attributes)
            graph_attention_model.train_model(trainer, train_loader, eval_loader, )
        except KeyError as e:
            logging.error(e)
        except ValueError as e:
            logging.error(e)
        except AssertionError as e:
            logging.error(f'Error: {str(e)}')
        except DatasetException as e:
            logging.error(f'Error: {str(e)}')
        except GraphException as e:
            logging.error(f'Error: {str(e)}')

    def __get_loader_attributes(self) -> Dict[AnyStr, Any]:
        return {
            'id': 'NeighborLoader',
            'num_neighbors': self.neighbors,
            'batch_size': 32,
            'replace': True,
            'num_workers': 4
        }

    def __get_model_attributes(self, title: AnyStr, dataset: Dataset) -> Dict[AnyStr, Any]:
        return {
            'model_id': title,
            'graph_attention_blocks': [
                {
                    'block_id': 'GAT_1',
                    'attention_layer': GATConv(in_channels=dataset[0].num_node_features,
                                               out_channels=self.hidden_channels,
                                               heads=self.num_heads),
                    'num_channels': self.hidden_channels,
                    'activation': None,
                    'batch_norm': None,
                    'dropout': 0.25
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'Output',
                    'in_features': self.hidden_channels * 4,
                    'out_features': dataset.num_classes,
                    'activation': None
                }
            ]
        }

    def __get_training_attributes(self, dataset: Dataset) -> Dict[AnyStr, Any]:
        # 2. Initialize the class weights for unbalanced data
        class_weights = GraphDataLoader.class_weights(dataset[0])
        return {
                'dataset_name': self.dataset_name,
                # Model training Hyperparameters
                'learning_rate': 0.0008,
                'weight_decay': 5e-4,
                'batch_size': 32,
                'loss_function': nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights),
                'momentum': 0.90,
                'encoding_len': -1,
                'train_eval_ratio': 0.9,
                'weight_initialization': 'kaiming',
                'optim_label': 'adam',
                'drop_out': 0.25,
                'is_class_imbalance': True,
                'class_weights': class_weights,
                'patience': 2,
                'min_diff_loss': 0.02,
                'epochs': 28,
                # Model configuration
                'hidden_channels': 64,
                # Performance metric definition
                'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR'],
                'plot_parameters': {
                    'count': 0,
                    'x_label': 'Epochs',
                    'title': f'Graph Attention - Sampling [6, 3] - {self.dataset_name}',
                    'x_label_size': 12,
                    'fig_size': (10, 8)
                }
            }


if __name__ == "__main__":
    graph_attention_model_play = GraphAttentionModelPlay(dataset_name='Cora',
                                                         neighbors=[6, 3],
                                                         hidden_channels=64,
                                                         num_heads=4)
    graph_attention_model_play.play()