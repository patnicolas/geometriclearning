import unittest
import logging
from typing import List, AnyStr, Any, Dict
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
            graph_SAGE_block_1 = GraphSAGEBlock[SAGEConv](block_id='SAGE 24-256',
                                                          graph_SAGE_layer=sage_conv_1,
                                                          batch_norm_module=BatchNorm(hidden_channels),
                                                          activation_module=nn.ReLU(),
                                                          dropout_module=nn.Dropout(0.2))

            sage_conv_2 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            graph_SAGE_block_2 = GraphSAGEBlock[SAGEConv](block_id='SAGE 256-256',
                                                          graph_SAGE_layer=sage_conv_2,
                                                          batch_norm_module=BatchNorm(hidden_channels),
                                                          activation_module=nn.ReLU(),
                                                          dropout_module=nn.Dropout(0.2))

            sage_conv_3 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            graph_SAGE_block_3 = GraphSAGEBlock[SAGEConv](block_id='Conv 256-8', graph_SAGE_layer=sage_conv_3)
            mlp_block = MLPBlock(block_id='Fully connected',
                                 layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                                 activation_module=None)

            graph_SAGE_model = GraphSAGEModel[SAGEConv](
                model_id='Flicker test dataset',
                graph_SAGE_blocks=[graph_SAGE_block_1, graph_SAGE_block_2, graph_SAGE_block_3],
                mlp_blocks=[mlp_block]
            )
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
                        'activation': None,
                        'dropout': 0.3
                    }
                ]
            }
            graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
            graph_SAGE_model = graph_SAGE_builder.build()
            graph_SAGE_model.reset_parameters()
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

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_training_cora(self):
        from deeplearning.block.graph import GraphException

        try:
            dataset_name = 'Cora'
            neighbors = [20, 12]
            num_layers = 4
            GraphSAGEModelTest.process_training(dataset_name, num_layers, neighbors)
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
    def process_training(dataset_name: AnyStr, num_layers: int, neighbors: List[int]):
        pyg_dataset = PyGDatasets(dataset_name)
        dataset = pyg_dataset()
        # Parameterization
        train_attrs, model_attrs = GraphSAGEModelTest.build_config(dataset_name=pyg_dataset.name,
                                                                   lr=0.0008,
                                                                   num_layers=num_layers,
                                                                   neighbors=neighbors,
                                                                   _dataset=dataset,
                                                                   hidden_channels=256,
                                                                   epochs=40)
        logging.info(f'\nTraining attributes\n{train_attrs}\nModel attributes:\n{model_attrs}')
        sampling_attrs = {
            'id': 'NeighborLoader',
            'num_neighbors': neighbors,
            'batch_size': 32,
            'replace': True,
            'num_workers': 4
        }
        GraphSAGEModelTest.execute_training(training_attributes=train_attrs,
                                            model_attributes=model_attrs,
                                            sampling_attrs=sampling_attrs)

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
                                                                       _dataset=_dataset,
                                                                       hidden_channels=40,
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

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_cora(self):
        from plots.plotter import PlotterParameters, Plotter

        auroc_6_3 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Cora_2layer_0.0008_random_[6, 3].json',
            'AuROC')
        auroc_12_8 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Cora_2layer_0.0008_random_[12, 8].json',
            'AuROC')
        auroc_12_12_6 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Cora_2layer_0.0008_random_[12, 12, 6].json',
            'AuROC')

        plotter_params = PlotterParameters(count=0,
                                           x_label='X',
                                           y_label='Y',
                                           title='Impact of Neighbors Sampling',
                                           fig_size=(12, 8))
        labels = ['AuROC-6.3', 'AuROC-12.8', 'AuROC-12.12.6']
        Plotter.plot([auroc_6_3, auroc_12_8, auroc_12_12_6], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_pubmed_auROC(self):
        from plots.plotter import PlotterParameters, Plotter

        auroc_6_3 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[6, 3].json',
            'AucROC')
        auroc_12_8 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[12, 8].json',
            'AucROC')
        auroc_12_12_6 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[12, 12, 6].json',
            'AucROC')

        plotter_params = PlotterParameters(count=0,
                                           x_label='Epochs',
                                           y_label='AuROC',
                                           title='Impact of Neighbors Sampling - PubMed',
                                           fig_size=(12, 8))
        labels = ['Neighbors Sampling 6 x 3', 'Neighbors Sampling 12 x 8', 'Neighbors Sampling 12 x 12 x 6']
        Plotter.plot([auroc_6_3, auroc_12_8, auroc_12_12_6], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_pubmed_f1(self):
        from plots.plotter import PlotterParameters, Plotter

        aucroc_6_3 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[6, 3].json',
            'F1')
        aucroc_12_8 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[12, 8].json',
            'F1')
        aucroc_12_12_6 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[12, 12, 6].json',
            'F1')

        plotter_params = PlotterParameters(count=0,
                                           x_label='Epochs',
                                           y_label='F1',
                                           title='Impact of Neighbors Sampling - PubMed',
                                           fig_size=(12, 8))
        labels = ['Neighbors Sampling 6 x 3', 'Neighbors Sampling 12 x 8', 'Neighbors Sampling 12 x 12 x 6']
        Plotter.plot([aucroc_6_3, aucroc_12_8, aucroc_12_12_6], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_pubmed_f1_2_3_layers_F1(self):
        from plots.plotter import PlotterParameters, Plotter

        aucroc_6_3_2 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[6, 3].json',
            'F1')
        aucroc_6_3_3 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_3layer_0.0012_random_[6, 3].json',
            'F1')

        plotter_params = PlotterParameters(count=0,
                                           x_label='Epochs',
                                           y_label='F1',
                                           title='Impact of Number SAGE Conv layers - PubMed',
                                           fig_size=(12, 8))
        labels = ['Neighbors Sampling 6 x 3 - 2 Conv', 'Neighbors Sampling 6 x 3 - 3 Conv']
        Plotter.plot([aucroc_6_3_2, aucroc_6_3_3], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_pubmed_2_3_layers_AuROC(self):
        from plots.plotter import PlotterParameters, Plotter

        auroc_6_3_2 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_2layer_0.0012_random_[6, 3].json',
            'AucROC')
        auroc_6_3_3 = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_PubMed_3layer_0.0012_random_[6, 3].json',
            'AucROC')

        plotter_params = PlotterParameters(count=0,
                                           x_label='Epochs',
                                           y_label='AuROC',
                                           title='Impact of Number SAGE Conv layers - PubMed',
                                           fig_size=(12, 8))
        labels = ['Neighbors Sampling 6 x 3 - 2 Conv', 'Neighbors Sampling 6 x 3 - 3 Conv']
        Plotter.plot([auroc_6_3_2, auroc_6_3_3], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_flickr_F1(self):
        from plots.plotter import PlotterParameters, Plotter

        flickr_12K = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Flickr_2layer_0.0018_random_[20, 12]_12Knodes.json',
            'F1')
        flickr_32K = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Flickr_2layer_0.0018_random_[20, 12]_32Knodes.json',
            'F1')
        flickr_89K = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Flickr_2layer_0.0018_random_[20, 12]_89Knodes.json',
            'F1')

        plotter_params = PlotterParameters(count=0,
                                           x_label='Epochs',
                                           y_label='F1',
                                           title='Impact of Graph Nodes sub-sampling - F1 - Flickr',
                                           fig_size=(12, 8))
        labels = ['Neighbors Sampling 20x12 - 2 Conv-12K',
                  'Neighbors Sampling 20x12 - 2 Conv-32K',
                  'Neighbors Sampling 20x12 - 2 Conv-89K']
        Plotter.plot([flickr_12K, flickr_32K, flickr_89K], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_flickr_AuROC(self):
        from plots.plotter import PlotterParameters, Plotter

        flickr_12K = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Flickr_2layer_0.0018_random_[20, 12]_12Knodes.json',
            'AucROC')
        flickr_32K = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Flickr_2layer_0.0018_random_[20, 12]_32Knodes.json',
            'AucROC')
        flickr_89K = GraphSAGEModelTest.load_data(
            '../../../output_plots/SAGE_Flickr_2layer_0.0018_random_[20, 12]_89Knodes.json',
            'AucROC')

        plotter_params = PlotterParameters(count=0,
                                           x_label='Epochs',
                                           y_label='AuROC',
                                           title='Impact of Graph Nodes sub-sampling - AuROC - Flickr',
                                           fig_size=(12, 8))
        labels = ['Neighbors Sampling 20x12 - 2 Conv-12K',
                  'Neighbors Sampling 20x12 - 2 Conv-32K',
                  'Neighbors Sampling 20x12 - 2 Conv-89K']
        Plotter.plot([flickr_12K, flickr_32K, flickr_89K], labels, plotter_params)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_analyze_data_flickr_latency(self):
        from plots.plotter import PlotterParameters, Plotter

        plotter_params = PlotterParameters(count=0,
                                           x_label='Number Graph Nodes',
                                           y_label='Seconds',
                                           title='Impact of Graph  on Latency - Flickr',
                                           fig_size=(12, 8))
        labels = ['Latency']
        latency = [1491, 8549, 31024]
        Plotter.plot([latency], labels, plotter_params)

    @staticmethod
    def load_data(filename: AnyStr, column: AnyStr) -> List[float]:
        import json

        with open(filename, 'r', encoding='utf-8') as f:
            s = json.load(f)
        return s[column]

    @staticmethod
    def build_config(dataset_name: AnyStr,
                     lr: float,
                     num_layers: int,
                     neighbors: List[int],
                     _dataset,
                     hidden_channels: int,
                     epochs: int) -> (Dict[AnyStr, Any], Dict[AnyStr, Any]):
        from dataset.graph.graph_data_loader import GraphDataLoader
        _data = _dataset[0]
        class_weights = GraphDataLoader.class_weights(_data)
        title = f'SAGE_{dataset_name}_NeighborLoader{neighbors}_{num_layers}layers'

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
            'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR']
        }

        model_attributes_2 = {
            'model_id': f'Graph{title}',
            'graph_SAGE_blocks': [
                {
                    'block_id': 'SAGE Layer 1',
                    'SAGE_layer': SAGEConv(in_channels=_dataset[0].num_node_features, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
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

        model_attributes_4 = {
            'model_id': f'Graph{title}',
            'graph_SAGE_blocks': [
                {
                    'block_id': 'SAGE Layer 1',
                    'SAGE_layer': SAGEConv(in_channels=_dataset[0].num_node_features, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
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

        model_attributes_6 = {
            'model_id': f'Graph{title}',
            'graph_SAGE_blocks': [
                {
                    'block_id': 'SAGE Layer 1',
                    'SAGE_layer': SAGEConv(in_channels=_dataset[0].num_node_features, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
                    'batch_norm': None,
                    'dropout': 0.25
                },
                {
                    'block_id': 'SAGE Layer 2',
                    'SAGE_layer': SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    'num_channels': hidden_channels,
                    'activation': nn.ReLU(),
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
        match num_layers:
            case 2: model_attributes = model_attributes_2
            case 4: model_attributes = model_attributes_4
            case 6: model_attributes = model_attributes_6
            case _: model_attributes = model_attributes_2
        return training_attributes, model_attributes

    @staticmethod
    def execute_training(training_attributes: Dict[AnyStr, Any],
                         model_attributes: Dict[AnyStr, Any],
                         sampling_attrs: Dict[AnyStr, Any],
                         num_subgraph_nodes: int = 0) -> None:
        from deeplearning.training.gnn_training import GNNTraining
        from dataset.graph.graph_data_loader import GraphDataLoader

        # Step 1:
        graph_SAGE_builder = GraphSAGEBuilder(model_attributes)
        graph_SAVE_model = graph_SAGE_builder.build()
        # Step 2:  Create the trainer using the training attributes dictionary
        trainer = GNNTraining.build(training_attributes)
        # Step 3: Create the data loader and extract a sub graph
        graph_data_loader = GraphDataLoader(dataset_name=training_attributes['dataset_name'],
                                            sampling_attributes=sampling_attrs,
                                            num_subgraph_nodes=num_subgraph_nodes)
        logging.info(graph_data_loader)
        train_loader, eval_loader = graph_data_loader()
        # Step 4: Train and Validate the model
        graph_SAVE_model.train_model(trainer, train_loader, eval_loader)
