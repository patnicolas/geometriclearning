import unittest

from mpmath import hyper

from dl import GraphException
from dataset import DatasetException
from dl.block.mlp_block import MLPBlock
from dl.training.hyper_params import HyperParams
from dl.training.gnn_training import GNNTraining
from dl.model.gnn_base_model import GNNBaseModel
from dataset.graph.graph_data_loader import GraphDataLoader
from metric.metric_type import MetricType
from metric.built_in_metric import BuiltInMetric
import torch.nn as nn
import os
from typing import Dict, Any, AnyStr
import logging
import os
import python
from python import SKIP_REASON


def show(attrs: Dict[AnyStr, Any]) -> AnyStr:
    return ', '.join([f'{k}:{v}' for k, v in attrs.items()])


class GNNTrainingTest(unittest.TestCase):
    import torch
    torch.set_default_dtype(torch.float32)

    def test_build(self):
        training_attributes = {
            'dataset_name': 'Flickr',
            # Model training Hyperparameters
            'learning_rate': 0.0005,
            'batch_size': 64,
            'loss_function': None,
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
            # Performance metric definition
            'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1'],
            'plot_parameters': [
                {'title': 'Accuracy', 'x_label': 'epoch', 'y_label': 'accuracy'},
                {'title': 'Precision', 'x_label': 'epochs', 'y_label': 'precision'},
                {'title': 'Recall', 'x_label': 'epochs', 'y_label': 'recall'},
                {'title': 'F1', 'x_label': 'epochs', 'y_label': 'F1'},
            ]
        }

        gnn_training = GNNTraining.build(training_attributes)
        logging.info(gnn_training)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_random_walk_loader(self):
        from torch_geometric.datasets.flickr import Flickr

        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]

            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'GraphSAINTRandomWalkSampler',
                'walk_length': 4,
                'num_steps': 16,
                'batch_size': 4096,
                'sample_coverage': 128
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)

            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.performance_metrics.performance_values[MetricType.Accuracy]
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_neighbor_loader_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]

            training_attributes = {
                'dataset_name': 'Flickr',
                # Model training Hyperparameters
                'learning_rate': 0.0005,
                'batch_size': 64,
                'loss_function': None,
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
                # Model configuration
                'hidden_channels': 256,
                # Performance metric definition
                'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1'],
                'plot_parameters': [
                    {'title': 'Accuracy', 'x_label': 'epoch', 'y_label': 'accuracy'},
                    {'title': 'Precision', 'x_label': 'epochs', 'y_label': 'precision'},
                    {'title': 'Recall', 'x_label': 'epochs', 'y_label': 'recall'},
                    {'title': 'F1', 'x_label': 'epochs', 'y_label': 'F1'},
                ]
            }

            attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': [12, 6, 3],
                'batch_size': 128,
                'replace': True,
                'num_workers': 4
            }
            network = GNNTraining.build(training_attributes)
            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()
            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)

        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_neighbor_loader_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': [6, 4],
                'batch_size': 1024,
                'replace': True
            }
            network = GNNTraining(hyper_params=hyper_parameters,
                                  metrics_attributes=metric_labels)

            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_random_loader_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'RandomNodeLoader',
                'num_parts': 256
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)

            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_random_loader_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'RandomNodeLoader',
                'num_parts': 256
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)

            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_graph_SAINT_node_sampler_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'GraphSAINTNodeSampler',
                'num_steps': 256,
                'sample_coverage': 100,
                'batch_size': 1024
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)

            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_shadow_khop_sampler(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'ShaDowKHopSampler',
                'depth': 3,
                'num_neighbors': 8,
                'batch_size': 1024
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)
            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_cluster_loader_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'ClusterLoader',
                'num_parts': 256,
                'recursive': False,
                'batch_size': 2048,
                'keep_inter_cluster_edges': False
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)
            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_cluster_loader_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = GNNTrainingTest.default_metrics_attributes()
            hyper_parameters = GNNTrainingTest.default_hyperparams(_dataset.num_classes)
            attrs = {
                'id': 'ClusterLoader',
                'num_parts': 256,
                'recursive': True,
                'batch_size': 2048,
                'keep_inter_cluster_edges': False
            }
            network = GNNTraining(hyper_params=hyper_parameters, metrics_attributes=metric_labels)
            gnn_base_model = GNNTrainingTest.create_model(num_node_features=_dataset.num_node_features,
                                                          num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            train_loader, eval_loader = graph_data_loader()

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GraphException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_draw_sample(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            attrs = {
                'id': 'ClusterLoader',
                'num_parts': 256,
                'recursive': True,
                'batch_size': 2048,
                'keep_inter_cluster_edges': False
            }
            graph_data_loader = GraphDataLoader(dataset_name='Flickr', sampling_attributes=attrs)
            graph_data_loader.draw_sample(
                first_node_index=10,
                last_node_index=26,
                node_color='blue',
                node_size=30,
                label=f'Flickr - ClusterLoader,num_parts:256,batch_size:2048,range:[10,25]')
            self.assertTrue(True)
        except DatasetException as e:
            logging.info(str(e))
            self.assertTrue(False)

    def test_draw_sample_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            attrs = {
                'id': 'GraphSAINTNodeSampler',
                'num_steps': 256,
                'sample_coverage': 100,
                'batch_size': 1024
            }
            graph_data_loader = GraphDataLoader(dataset_name='Flickr',
                                                sampling_attributes=attrs,
                                                num_subgraph_nodes=24,
                                                start_index=6)
            logging.info(f'Number of nodes {_data.num_nodes}')
            self.assertTrue(_data.num_nodes > 0)
        except DatasetException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_compare_accuracy(self):
        from plots.plotter import PlotterParameters, Plotter

        try:
            file_neighbor_loader = 'Flickr: NeighborLoader,neighbors:[6, 4],batch:1024'
            neighbor_loader_dict = TrainingMonitor.load_summary(TrainingMonitor.output_folder, file_neighbor_loader)
            accuracy_neighbor = [float(x) for x in neighbor_loader_dict['Accuracy']]

            file_random_loader = 'Flickr: RandomNodeLoader,num_parts=256'
            random_loader_dict = TrainingMonitor.load_summary(TrainingMonitor.output_folder, file_random_loader)
            accuracy_random = [float(x) for x in random_loader_dict['Accuracy']]

            file_graph_saint_random_walk_loader = 'GraphSAINTRandomWalkSampler,walk_length:3,steps:12,batch:4096'
            graph_saint_random_walk_dict = TrainingMonitor.load_summary(TrainingMonitor.output_folder,
                                                                        file_graph_saint_random_walk_loader)
            accuracy_graph_saint_random_walk = [float(x) for x in graph_saint_random_walk_dict['Accuracy']]
            plotter_params = PlotterParameters(0,
                                               x_label='epochs',
                                               y_label='Accuracy',
                                               title='Comparison Accuracy Graph Data Loader',
                                               fig_size=(11, 8))
            Plotter.plot(values=[accuracy_neighbor[0:40], accuracy_random[0:40], accuracy_graph_saint_random_walk[0:40]],
                         labels=['NeighborLoader', 'RandomLoader', 'GraphsSAINTRandomWalk'],
                         plotter_parameters=plotter_params)
            self.assertTrue(True)
        except DatasetException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_compare_precision(self):
        from plots.plotter import PlotterParameters, Plotter

        file_neighbor_loader = 'Flickr: NeighborLoader,neighbors:[6, 4],batch:1024'
        neighbor_loader_dict = TrainingMonitor.load_summary(TrainingMonitor.output_folder, file_neighbor_loader)
        precision_neighbor = [float(x) for x in neighbor_loader_dict['Precision']]

        file_random_loader = 'Flickr: RandomNodeLoader,num_parts=256'
        random_loader_dict = TrainingMonitor.load_summary(TrainingMonitor.output_folder, file_random_loader)
        precision_random = [float(x) for x in random_loader_dict['Precision']]

        file_graph_saint_random_walk_loader = 'Flickr: GraphSAINTRandomWalkSampler,walk_length:3,steps:12,batch:4096'
        graph_saint_random_walk_dict = TrainingMonitor.load_summary(TrainingMonitor.output_folder,
                                                                    file_graph_saint_random_walk_loader)
        precision_graph_saint_random_walk = [float(x) for x in graph_saint_random_walk_dict['Precision']]
        plotter_params = PlotterParameters(0,
                                           x_label='epochs',
                                           y_label='Precision',
                                           title='Comparison Precision Graph Data Loader',
                                           fig_size=(11, 8))
        Plotter.plot(values=[precision_neighbor[0:60], precision_random[0:60], precision_graph_saint_random_walk[0:60]],
                     labels=['NeighborLoader', 'RandomLoader', 'GraphsSAINTRandomWalk'],
                     plotter_parameters=plotter_params)

    """ --------------------------  Supporting methods --------------------  """

    @staticmethod
    def default_metrics_attributes() -> Dict[MetricType, BuiltInMetric]:
        return {
            BuiltInMetric.accuracy_label: BuiltInMetric(MetricType.Accuracy, encoding_len=-1, is_weighted=True),
            BuiltInMetric.precision_label: BuiltInMetric(MetricType.Precision, encoding_len=-1, is_weighted=True),
            BuiltInMetric.recall_label: BuiltInMetric(MetricType.Recall, encoding_len=-1, is_weighted=True)
        }

    @staticmethod
    def default_hyperparams(num_classes: int) -> HyperParams:
        return HyperParams(
            lr=0.0005,
            momentum=0.90,
            epochs=8,
            optim_label='adam',
            batch_size=128,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            encoding_len=num_classes)

    @staticmethod
    def create_model(num_node_features: int, num_classes: int) -> GNNBaseModel:
        from torch_geometric.nn import GraphConv
        from dl.block.graph.g_message_passing_block import GMessagePassingBlock
        from dl.model.gnn_base_model import GNNBaseModel

        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gnn_block_1 = GMessagePassingBlock(block_id='K1',
                                           message_passing_module=conv_1,
                                           activation_module=nn.ReLU(),
                                           drop_out_module=nn.Dropout(0.2))
        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gnn_block_2 = GMessagePassingBlock(block_id='K2',
                                           message_passing_module=conv_2,
                                           activation_module=nn.ReLU(),
                                           drop_out_module=nn.Dropout(0.2))
        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gnn_block_3 = GMessagePassingBlock(block_id='K3',
                                           message_passing_module=conv_3,
                                           activation_module=nn.ReLU(),
                                           drop_out_module=nn.Dropout(0.2))

        ffnn_block = MLPBlock(block_id='Output',
                              layer_module=nn.Linear(3*hidden_channels, num_classes),
                              activation_module=nn.LogSoftmax(dim=-1))
        return GNNBaseModel(model_id='Flickr',
                            gnn_blocks=[gnn_block_1, gnn_block_2, gnn_block_3],
                            mlp_blocks=[ffnn_block])
