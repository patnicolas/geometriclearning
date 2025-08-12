import unittest
import logging
from typing import List, AnyStr, Any, Dict, Optional
from deeplearning.block.graph.graph_sage_block import GraphSAGEBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.graph.graph_sage_model import GraphSAGEModel, GraphSAGEBuilder
from torch_geometric.nn import SAGEConv, BatchNorm
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.training import TrainingException
import torch_geometric
import torch.nn as nn
import torch
from dataset import DatasetException
import os
from python import SKIP_REASON


class GraphSAGEModelTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_1(self):
        try:
            import torch_geometric
            from dataset.graph.pyg_datasets import PyGDatasets

            hidden_channels = 256
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()
            _data: torch_geometric.data.Data = _dataset[0]

            sage_conv_1 = SAGEConv(in_channels=_data.num_node_features, out_channels=hidden_channels)
            graph_SAGE_block_1 = GraphSAGEBlock(block_id='SAGE 24-256',
                                                graph_SAGE_layer=sage_conv_1,
                                                batch_norm_module=BatchNorm(hidden_channels),
                                                activation_module=nn.ReLU(),
                                                dropout_module=nn.Dropout(0.2))

            sage_conv_2 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            graph_SAGE_block_2 = GraphSAGEBlock(block_id='SAGE 256-256',
                                                graph_SAGE_layer=sage_conv_2,
                                                batch_norm_module=BatchNorm(hidden_channels),
                                                activation_module=nn.ReLU(),
                                                dropout_module=nn.Dropout(0.2))

            sage_conv_3 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            graph_SAGE_block_3 = GraphSAGEBlock(block_id='Conv 256-8', graph_SAGE_layer=sage_conv_3)
            mlp_block = MLPBlock(block_id='Fully connected',
                                 layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                                 activation_module=None)

            graph_SAGE_model = GraphSAGEModel(model_id='Flicker test dataset',
                                              graph_SAGE_blocks=[graph_SAGE_block_1, graph_SAGE_block_2, graph_SAGE_block_3],
                                              mlp_blocks=[mlp_block])
            logging.info(f'\n{graph_SAGE_model}')
            params = list(graph_SAGE_model.parameters())
            logging.info(f'\nParameters:\n{params}')
            self.assertTrue(len(params) == 15)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_build(self):
        try:
            out_channels = 256
            num_node_features = 64
            in_features = 256
            out_features = 8

            model_attributes = {
                'model_id': 'MyModel',
                'graph_SAGE_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'SAGE_layer': SAGEConv(in_channels=num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'SAGE_layer': SAGEConv(in_channels=out_channels, out_channels=out_channels),
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
            graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
            graph_SAGE_model = graph_SAGE_builder.build()
            logging.info(graph_SAGE_model)
            params = list(graph_SAGE_model.parameters())
            self.assertTrue(len(params) == 12)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
                'graph_SAGE_blocks': [
                    {
                        'block_id': 'MyBlock_1',
                        'SAGE_layer': SAGEConv(in_channels=_data.num_node_features, out_channels=out_channels),
                        'num_channels': out_channels,
                        'activation': nn.ReLU(),
                        'batch_norm': BatchNorm(out_channels),
                        'dropout': 0.25
                    },
                    {
                        'block_id': 'MyBlock_2',
                        'SAGE_layer': SAGEConv(in_channels=out_channels, out_channels=out_channels),
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
            graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
            graph_SAGE_model = graph_SAGE_builder.build()
            logging.info(graph_SAGE_model)
            graph_SAGE_model.forward(data=_data)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except DatasetException as e:
            logging.error(e)
            self.assertTrue(False)

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_training_flickr(self):
        import os
        logging.info(os.getcwd())
        from torch_geometric.datasets.flickr import Flickr
        from deeplearning.training.gnn_training import GNNTraining
        from dataset.graph.graph_data_loader import GraphDataLoader
        from deeplearning.block.graph import GraphException

        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)

            # Parameterization
            neighbors = [10, 8]
            training_attributes, model_attributes = GraphSAGEModelTest.create_configuration(dataset_name='Flickr',
                                                                                            lr=0.0005,
                                                                                            neighbors=neighbors,
                                                                                            hidden_channels=64,
                                                                                            epochs=4,
                                                                                            _dataset=_dataset)
            sampling_attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }
            # Step 1: Create the SAGE model
            graph_conv_builder = GraphSAGEBuilder(model_attributes)
            graph_conv_model = graph_conv_builder.build()
            # Step 2:  Create the trainer
            trainer = GNNTraining.build(training_attributes)
            # Step 3: Create the data loader
            graph_data_loader = GraphDataLoader(dataset_name='Flickr',
                                                sampling_attributes=sampling_attrs,
                                                num_subgraph_nodes=None)
            logging.info(graph_data_loader)
            train_loader, eval_loader = graph_data_loader()

            # Step 4: Train and Validate the model
            graph_conv_model.train_model(trainer, train_loader, eval_loader)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except AssertionError as e:
            logging.info(f'Assertion: {str(e)}')
            self.assertTrue(False)
        except TrainingException as e:
            logging.info(f'Training: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Dataset: {str(e)}')
            self.assertTrue(False)
        except GraphException as e:
            logging.info(f'Graph model: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_training_cora(self):
        from torch_geometric.datasets import Planetoid
        from deeplearning.block.graph import GraphException

        try:
            _dataset = Planetoid(root='/tmp/Cora', name='Cora')
            # Parameterization
            neighbors = [4, 4]
            training_attributes, model_attributes = GraphSAGEModelTest.create_configuration(dataset_name='Cora',
                                                                                            lr=0.0008,
                                                                                            neighbors=neighbors,
                                                                                            hidden_channels=32,
                                                                                            _dataset=_dataset,
                                                                                            epochs=70)
            sampling_attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }
            GraphSAGEModelTest.execute_training(dataset_name='Cora',
                                                training_attributes=training_attributes,
                                                model_attributes=model_attributes,
                                                sampling_attrs=sampling_attrs,
                                                num_subgraph_nodes=None)
        except KeyError as e:
            logging.error(e)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)
        except AssertionError as e:
            logging.info(f'Assertion: {str(e)}')
            self.assertTrue(False)
        except TrainingException as e:
            logging.info(f'Training: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Dataset: {str(e)}')
            self.assertTrue(False)
        except GraphException as e:
            logging.info(f'Graph model: {str(e)}')
            self.assertTrue(False)

    @staticmethod
    def create_configuration(dataset_name: AnyStr,
                             lr: float,
                             neighbors: List[int],
                             _dataset,
                             hidden_channels: int,
                             epochs: int) -> (Dict[AnyStr, Any], Dict[AnyStr, Any]):
        from dataset.graph.graph_data_loader import GraphDataLoader
        _data = _dataset[0]
        class_weights = GraphDataLoader.class_weights(_data)
        title = f'SAGE_{dataset_name}_{lr}_{neighbors}'

        # Parameterization
        training_attributes = {
            'dataset_name': 'Cora',
            # Model training Hyperparameters
            'learning_rate': lr,
            'batch_size': 32,
            'loss_function': nn.CrossEntropyLoss(),
            'momentum': 0.95,
            'weight_decay': 1e-3,
            'encoding_len': -1,
            'train_eval_ratio': 0.9,
            'weight_initialization': 'xavier',
            'optim_label': 'adam',
            'drop_out': 0.25,
            'is_class_imbalance': True,
            'class_weights': class_weights,
            'patience': 2,
            'min_diff_loss': 0.02,
            'epochs': epochs,
            # Model configuration
            'hidden_channels': 32,
            # Performance metric definition
            'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AucROC', 'AucPR'],
            'plot_parameters': {
                'x_label': 'epochs',
                'title': title,
                'x_label_size': 11,
                'fig_size': (13, 8),
                'plot_filename': f'../../../output_plots/{title}'
            }
        }
        model_attributes = {
            'model_id': f'GraphSAGE_Cora_{lr}_{neighbors}',
            'graph_SAGE_blocks': [
                {
                    'block_id': 'SAGE Layer 1',
                    'SAGE_layer': SAGEConv(in_channels=_dataset[0].num_node_features, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    # 'batch_norm': BatchNorm(hidden_channels),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    # 'batch_norm': BatchNorm(hidden_channels),
                    'batch_norm': None,
                    'dropout': 0.25
                }
            ],
            'mlp_blocks': [
                {
                    'block_id': 'Node classification block',
                    'in_features': hidden_channels,
                    'out_features': _dataset.num_classes,
                    'activation': None
                }
            ]
        }
        return training_attributes, model_attributes

    @staticmethod
    def execute_training(dataset_name: AnyStr,
                         training_attributes: Dict[AnyStr, Any],
                         model_attributes: Dict[AnyStr, Any],
                         sampling_attrs: Dict[AnyStr, Any],
                         num_subgraph_nodes: Optional[int] = None) -> None:
        from deeplearning.training.gnn_training import GNNTraining
        from dataset.graph.graph_data_loader import GraphDataLoader

        graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
        graph_SAVE_model = graph_SAGE_builder.build()
        # Step 2:  Create the trainer
        trainer = GNNTraining.build(training_attributes)
        # Step 3: Create the data loader
        graph_data_loader = GraphDataLoader(dataset_name=dataset_name,
                                            sampling_attributes=sampling_attrs,
                                            num_subgraph_nodes=num_subgraph_nodes)
        logging.info(graph_data_loader)
        train_loader, eval_loader = graph_data_loader()
        # Step 4: Train and Validate the model
        graph_SAVE_model.train_model(trainer, train_loader, eval_loader)
