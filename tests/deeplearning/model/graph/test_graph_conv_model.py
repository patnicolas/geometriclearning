import unittest
import logging
from deeplearning.block.graph.graph_conv_block import GraphConvBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.graph.graph_conv_model import GraphConvModel, GraphConvBuilder
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.nn.pool import TopKPooling
from dataset.graph.pyg_datasets import PyGDatasets
import torch_geometric
from dataset import DatasetException
import torch.nn as nn
import os
from python import SKIP_REASON


class GraphConvModelTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_1(self):
        hidden_channels = 256
        try:
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            graph_conv_block_1 = GraphConvBlock[GraphConv, TopKPooling](
                block_id='Conv 24-256',
                graph_conv_layer=GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels),
                batch_norm_module=BatchNorm(hidden_channels),
                activation_module=nn.ReLU(),
                pooling_module=TopKPooling(hidden_channels, ratio=0.4),
                dropout_module=nn.Dropout(0.2)
            )
            graph_conv_block_2 = GraphConvBlock[GraphConv, TopKPooling](
                block_id='Conv 256-256',
                graph_conv_layer= GraphConv(in_channels=hidden_channels, out_channels=hidden_channels),
                batch_norm_module=BatchNorm(hidden_channels),
                activation_module=nn.ReLU(),
                pooling_module=TopKPooling(hidden_channels, ratio=0.4),
                dropout_module=nn.Dropout(0.2)
            )
            graph_conv_block_3 = GraphConvBlock[GraphConv, TopKPooling](
                block_id='Conv 256-8',
                graph_conv_layer=GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
            )
            mlp_block = MLPBlock(block_id='Fully connected',
                                 layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                                 activation_module=nn.Softmax(dim=-1))

            graph_conv_model = GraphConvModel[GraphConv, TopKPooling](
                model_id='Flicker test dataset',
                graph_conv_blocks=[graph_conv_block_1, graph_conv_block_2, graph_conv_block_3],
                mlp_blocks=[mlp_block]
            )
            logging.info(f'\n{graph_conv_model}')
            params = list(graph_conv_model.parameters())
            logging.info(f'\nParameters:\n{params}')
            self.assertTrue(len(params) == 17)
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
                'graph_conv_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'conv_layer': GraphConv(in_channels=_data.num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'pooling': None,
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'conv_layer': GraphConv(in_channels=out_channels, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'pooling': None,
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
            graph_conv_builder = GraphConvBuilder(model_attributes)
            graph_conv_model = graph_conv_builder.build()
            logging.info(graph_conv_model)
            params = list(graph_conv_model.parameters())
            self.assertTrue(len(params) == 12)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except DatasetException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_build_2(self):
        out_channels = 256
        in_features = 256
        out_features = 8

        try:
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            model_attributes = {
                'model_id': 'MyModel',
                'graph_conv_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'conv_layer': GraphConv(in_channels=_data.num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'pooling': TopKPooling(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'conv_layer': GraphConv(in_channels=out_channels, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'pooling': TopKPooling(out_channels),
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
            graph_conv_builder = GraphConvBuilder(model_attributes)
            graph_conv_model = graph_conv_builder.build()
            logging.info(graph_conv_model)

            params = list(graph_conv_model.parameters())
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

    @unittest.skip('Ignore')
    def test_forward(self):
        out_channels = 256
        in_features = 256
        out_features = 8

        try:
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            model_attributes = {
                'model_id': 'MyModel',
                'graph_conv_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'conv_layer': GraphConv(in_channels=_data.num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'pooling': TopKPooling(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'conv_layer': GraphConv(in_channels=out_channels, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'pooling': TopKPooling(out_channels),
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
            graph_conv_builder = GraphConvBuilder(model_attributes)
            graph_conv_model = graph_conv_builder.build()
            logging.info(graph_conv_model)
            graph_conv_model.forward(data=_data)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except DatasetException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_training(self):
        from torch_geometric.datasets.flickr import Flickr
        from deeplearning.training.gnn_training import GNNTraining
        from dataset.graph.graph_data_loader import GraphDataLoader
        from deeplearning.block.graph import GraphException

        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]

            training_attributes = {
                'dataset_name': 'Flickr',
                # Model training Hyperparameters
                'learning_rate': 0.0005,
                'weight_decay': 1e-3,
                'batch_size': 32,
                'loss_function': nn.CrossEntropyLoss(),
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
                'epochs': 8,
                # Model configuration
                'hidden_channels': 64,
                # Performance metric definition
                'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1'],
                'plot_parameters': {
                    'title': 'Accuracy',
                    'x_label': 'epoch',
                    'x_label_size': 11,
                    'y_label': 'accuracy',
                    'fig_size': (13, 8),
                    'plot_filename': f'../../../output_plots/test1'
                }
            }
            attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': [12, 6, 3],
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }
            hidden_channels = 64
            in_features = 64
            model_attributes = {
                'model_id': 'MyModel',
                'graph_conv_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'conv_layer': GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels),
                        'num_channels': hidden_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(hidden_channels),
                        'pooling': None,
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'conv_layer': GraphConv(in_channels=hidden_channels, out_channels=hidden_channels),
                        'num_channels': hidden_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(hidden_channels),
                        'pooling': None,
                        'dropout': 0.25
                    }
                ],
                'mlp_blocks': [
                    {
                        'block_id': 'Classifier',
                        'in_features': in_features,
                        'out_features': _dataset.num_classes,
                        'activation': None,
                        'dropout': 0.3
                    }
                ]
            }
            graph_conv_builder = GraphConvBuilder(model_attributes)
            graph_conv_model = graph_conv_builder.build()

            trainer = GNNTraining.build(training_attributes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()
            graph_conv_model.train_model(trainer, train_loader, eval_loader)
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
