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
from dataset import DatasetException
import os
from python import SKIP_REASON, logger


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
            logger.error(e)
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
                        'dropout': 0.25
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
        from deeplearning.block.graph import GraphException
        import time

        try:
            pyg_dataset = PyGDatasets('Flickr')
            _dataset = pyg_dataset()

            # Parameterization
            neighbors = [20, 12]
            train_attrs, model_attrs = GraphSAGEModelTest.build_config(dataset_name=pyg_dataset.name,
                                                                       lr=0.0018,
                                                                       neighbors=neighbors,
                                                                       hidden_channels=32,
                                                                       epochs=48,
                                                                       _dataset=_dataset)
            sampling_attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }
            start = time.time()
            GraphSAGEModelTest.execute_training(training_attributes=train_attrs,
                                                model_attributes=model_attrs,
                                                sampling_attrs=sampling_attrs,
                                                num_subgraph_nodes=12000)
            duration = '{:.2f}'.format(time.time() - start)
            logging.info(f'\n>>>>>> Duration: {duration=} secs.')
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
        from deeplearning.block.graph import GraphException

        try:
            pyg_dataset = PyGDatasets('Cora')
            _dataset = pyg_dataset()
            # Parameterization
            neighbors = [6, 3]
            train_attrs, model_attrs = GraphSAGEModelTest.build_config(dataset_name=pyg_dataset.name,
                                                                       lr=0.0008,
                                                                       neighbors=neighbors,
                                                                       hidden_channels=32,
                                                                       _dataset=_dataset,
                                                                       epochs=90)
            sampling_attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }

            GraphSAGEModelTest.execute_training(training_attributes=train_attrs,
                                                model_attributes=model_attrs,
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

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_training_pubmed(self):
        from deeplearning.block.graph import GraphException

        try:
            pyg_dataset = PyGDatasets('PubMed')
            _dataset = pyg_dataset()
            # Parameterization
            neighbors = [12, 8]
            train_attrs, model_attrs = GraphSAGEModelTest.build_config(dataset_name=pyg_dataset.name,
                                                                       lr=0.0008,
                                                                       neighbors=neighbors,
                                                                       hidden_channels=40,
                                                                       _dataset=_dataset,
                                                                       epochs=90)
            sampling_attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            }
            GraphSAGEModelTest.execute_training(training_attributes=train_attrs,
                                                model_attributes=model_attrs,
                                                sampling_attrs=sampling_attrs,
                                                num_subgraph_nodes=16000)
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
    def build_config(dataset_name: AnyStr,
                     lr: float,
                     neighbors: List[int],
                     _dataset,
                     hidden_channels: int,
                     epochs: int) -> (Dict[AnyStr, Any], Dict[AnyStr, Any]):
        from dataset.graph.graph_data_loader import GraphDataLoader
        _data = _dataset[0]
        class_weights = GraphDataLoader.class_weights(_data)
        title = f'SAGE_{dataset_name}_2layer_{lr}_random_{neighbors}_12Knodes'

        # Parameterization
        training_attributes = {
            'dataset_name': dataset_name,
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
            'model_id': f'Graph{title}',
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
    def execute_training(training_attributes: Dict[AnyStr, Any],
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
        graph_data_loader = GraphDataLoader(dataset_name=training_attributes['dataset_name'],
                                            sampling_attributes=sampling_attrs,
                                            num_subgraph_nodes=num_subgraph_nodes)
        logging.info(graph_data_loader)
        train_loader, eval_loader = graph_data_loader()
        # Step 4: Train and Validate the model
        graph_SAVE_model.train_model(trainer, train_loader, eval_loader)
