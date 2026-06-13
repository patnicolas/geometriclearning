import unittest
import logging
from typing import List, AnyStr
from deeplearning.block.graph.graph_attention_block import GraphAttentionBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.graph.graph_attention_model import GraphAttentionModel, GraphAttentionBuilder
from torch_geometric.nn import GATConv, BatchNorm
from dataset.graph.pyg_datasets import PyGDatasets
import torch_geometric
from dataset import DatasetException
import torch.nn as nn
import os
from python import SKIP_REASON


class GraphAttentionModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init_1(self):
        hidden_channels = 256
        try:
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            graph_attention_block_1 = GraphAttentionBlock[GATConv](
                block_id='Attention 24-256',
                attention_layer=GATConv(in_channels=_data.num_node_features, out_channels=hidden_channels),
                batch_norm_module=BatchNorm(hidden_channels),
                activation_module=nn.ReLU(),
                dropout_module=nn.Dropout(0.2)
            )
            graph_attention_block_2 = GraphAttentionBlock[GATConv](
                block_id='Attention 256-256',
                attention_layer=GATConv(in_channels=hidden_channels, out_channels=hidden_channels),
                batch_norm_module=BatchNorm(hidden_channels),
                activation_module=nn.ReLU(),
                dropout_module=nn.Dropout(0.2)
            )
            graph_attention_block_3 = GraphAttentionBlock[GATConv](
                block_id='Conv 256-8',
                attention_layer=GATConv(in_channels=hidden_channels, out_channels=hidden_channels)
            )
            mlp_block = MLPBlock(block_id='Fully connected',
                                 layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                                 activation_module=nn.Softmax(dim=-1))

            graph_attention_model = GraphAttentionModel[GATConv](
                model_id='Flicker test dataset',
                graph_attention_blocks=tuple([graph_attention_block_1,
                                              graph_attention_block_2,
                                              graph_attention_block_3]),
                mlp_blocks=tuple([mlp_block])
            )
            graph_attention_model.reset_parameters()
            logging.info(f'\n{graph_attention_model}')
            params = list(graph_attention_model.parameters())
            logging.info(f'\nAttention Parameters:\n{params}')
            self.assertTrue(len(params) == 18)
        except DatasetException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_build(self):
        try:
            out_channels = 256
            in_features = 256
            out_features = 8
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            model_attributes = {
                'model_id': 'MyModel',
                'graph_attention_blocks': [
                    {
                        'block_id': 'GAT_1',
                        'attention_layer': GATConv(in_channels=_data.num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'GAT_2',
                        'attention_layer': GATConv(in_channels=out_channels, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'dropout': 0.25
                    }
                ],
                'mlp_blocks': [
                    {
                        'block_id': 'MyMLP',
                        'in_features': in_features,
                        'out_features': out_features,
                        'activation': nn.ReLU(),
                        'dropout': 0.3
                    }
                ]
            }
            graph_attention_builder = GraphAttentionBuilder(model_attributes)
            graph_attention_model = graph_attention_builder.build()
            graph_attention_model.reset_parameters()
            logging.info(graph_attention_model)
            params = list(graph_attention_model.parameters())
            self.assertTrue(len(params) == 14)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except DatasetException as e:
            logging.error(e)
            self.assertTrue(False)

    # @unittest.skip('Ignore')
    def test_training(self):
        from deeplearning.block.graph import GraphException

        try:
            dataset_name = 'Cora'
            neighbors = [6, 3]
            GraphAttentionModelTest.execute_training(dataset_name, neighbors)
            """
            num_layers = 4
            GraphAttentionModelTest.execute_training(dataset_name, neighbors, num_layers)

            neighbors = [20, 12]
            num_layers = 2
            GraphAttentionModelTest.execute_training(dataset_name, neighbors, num_layers)
            """
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except AssertionError as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @staticmethod
    def execute_training(dataset_name: AnyStr, neighbors: List[int]):
        from deeplearning.training.gnn_training import GNNTraining
        from dataset.graph.graph_data_loader import GraphDataLoader

        pyg_dataset = PyGDatasets(dataset_name)
        dataset = pyg_dataset()
        _data = dataset[0]

        class_weights = GraphDataLoader.class_weights(_data)
        title = f'Graph_Attention_{dataset_name}_Neighbors{neighbors}'

        training_attributes = {
                'dataset_name': 'Cora',
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
                    'title': 'Graph Attention - Sampling [6, 3] - Cora',
                    'x_label_size': 12,
                    'fig_size': (10, 8)
                }
            }
        loader_attributes = {
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }

        hidden_channels = 64

        model_attributes = {
            'model_id': title,
            'graph_attention_blocks': [
                {
                    'block_id': 'GAT_1',
                    'attention_layer': GATConv(in_channels=_data.num_node_features,
                                               out_channels=hidden_channels,
                                               heads=4),
                    'num_channels': hidden_channels,
                    'activation': None,
                    'batch_norm': None,
                    'dropout': 0.25
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'Output',
                    'in_features': hidden_channels*4,
                    'out_features': dataset.num_classes,
                    'activation': None
                }
            ]
        }

        graph_attention_builder = GraphAttentionBuilder(model_attributes)
        graph_attention_model = graph_attention_builder.build()

        trainer = GNNTraining.build(training_attributes)
        graph_data_loader = GraphDataLoader(dataset_name=dataset_name, sampling_attributes=loader_attributes)
        train_loader, eval_loader = graph_data_loader()
        graph_attention_model.train_model(trainer, train_loader, eval_loader, )

