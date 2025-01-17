import unittest

from dl import GNNException
from dataset import DatasetException
from dl.block.ffnn_block import FFNNBlock
from dl.training.hyper_params import HyperParams
from dl.training.gnn_training import GNNTraining
from dl.model.gnn_base_model import GNNBaseModel
from dataset.graph_data_loader import GraphDataLoader
from dl.training.training_summary import TrainingSummary
from metric.metric import Metric
import torch.nn as nn
import os


class GNNTrainingTest(unittest.TestCase):
    import torch
    torch.set_default_dtype(torch.float32)

    @unittest.skip('Ignore')
    def test_train_random_walk_loader(self):
        from torch_geometric.datasets.flickr import Flickr

        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=60,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'GraphSAINTRandomWalkSampler',
                'walk_length': 4,
                'num_steps': 16,
                'batch_size': 4096,
                'sample_coverage': 128
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='GraphSAINTRandomWalkSampler,walk_length:3,steps:12,batch:4096')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_neighbor_loader_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=48,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': [8, 8],
                'batch_size': 1024,
                'replace': True
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='NeighborLoader,neighbors:[8, 8],batch:1024')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_neighbor_loader_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=60,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'NeighborLoader',
                'num_neighbors': [6, 4],
                'batch_size': 1024,
                'replace': True
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='NeighborLoader,neighbors:[6, 4],batch:1024')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_random_loader_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=60,
                optim_label='adam',
                batch_size=256,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'RandomNodeLoader',
                'num_parts':256
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='RandomNodeLoader,num_parts=256')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_random_loader_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=60,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'RandomNodeLoader',
                'num_parts': 256
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='RandomNodeLoader,num_parts=256')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_graph_SAINT_node_sampler_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0008,
                momentum=0.90,
                epochs=60,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'GraphSAINTNodeSampler',
                'num_steps': 256,
                'sample_coverage': 100,
                'batch_size': 1024
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='GraphSAINTNodeSampler,num_parts=256,batch_size=1024,sample_coverage=100')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_shadow_khop_sampler(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=4,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'ShaDowKHopSampler',
                'depth': 3,
                'num_neighbors': 8,
                'batch_size': 1024
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='ShaDowKHopSampler,num_neighbors=8,batch_size=1024,depth=3')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_cluster_loader_1(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=8,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'ClusterLoader',
                'num_parts': 256,
                'recursive': False,
                'batch_size': 2048,
                'keep_inter_cluster_edges': False
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='ClusterLoader,num_parts=256,non-recursive,batch_size=2048')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_train_cluster_loader_2(self):
        from torch_geometric.datasets.flickr import Flickr
        try:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
            _dataset = Flickr(path)
            _data = _dataset[0]
            metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
            hyper_parameters = HyperParams(
                lr=0.0005,
                momentum=0.90,
                epochs=8,
                optim_label='adam',
                batch_size=128,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.2,
                train_eval_ratio=0.9,
                encoding_len=_dataset.num_classes)

            attrs = {
                'id': 'ClusterLoader',
                'num_parts': 256,
                'recursive': True,
                'batch_size': 2048,
                'keep_inter_cluster_edges': False
            }
            network = GNNTraining.build(hyper_params=hyper_parameters,
                                        metric_labels=metric_labels,
                                        title_attribute='ClusterLoader,num_parts=256,recursive,batch_size=2048')

            gnn_base_model = GNNTrainingTest.build(num_node_features=_dataset.num_node_features,
                                                   num_classes=_dataset.num_classes)
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            train_loader, eval_loader = graph_data_loader(num_workers=4)

            network.train(gnn_base_model.model_id, gnn_base_model, train_loader, eval_loader)
            accuracy_list = network.training_summary.metrics['Accuracy']
            self.assertTrue(len(accuracy_list) > 1)
            self.assertTrue(accuracy_list[-1].float() > 0.2)
        except GNNException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertTrue(False)

    @unittest.skip('Ignore')
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
            graph_data_loader = GraphDataLoader(loader_attributes=attrs, data=_data)
            graph_data_loader.draw_sample(first_node_index=10, last_node_index=24)
            self.assertTrue(True)
        except DatasetException as e:
            print(str(e))
            self.assertTrue(False)


    def test_compare_accuracy(self):
        from plots.plotter import PlotterParameters, Plotter

        file_neighbor_loader = 'Flickr: NeighborLoader,neighbors:[6, 4],batch:1024'
        neighbor_loader_dict = TrainingSummary.load_summary(TrainingSummary.output_folder, file_neighbor_loader)
        accuracy_neighbor = [float(x) for x in neighbor_loader_dict['Accuracy']]

        file_random_loader = 'Flickr: RandomNodeLoader,num_parts=256'
        random_loader_dict = TrainingSummary.load_summary(TrainingSummary.output_folder, file_random_loader)
        accuracy_random = [float(x) for x in random_loader_dict['Accuracy']]

        file_graph_saint_random_walk_loader = 'Flickr: GraphSAINTRandomWalkSampler,walk_length:3,steps:12,batch:4096'
        graph_saint_random_walk_dict = TrainingSummary.load_summary(TrainingSummary.output_folder,
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

    @unittest.skip('Ignore')
    def test_compare_precision(self):
        from plots.plotter import PlotterParameters, Plotter

        file_neighbor_loader = 'Flickr: NeighborLoader,neighbors:[6, 4],batch:1024'
        neighbor_loader_dict = TrainingSummary.load_summary(TrainingSummary.output_folder, file_neighbor_loader)
        precision_neighbor = [float(x) for x in neighbor_loader_dict['Precision']]

        file_random_loader = 'Flickr: RandomNodeLoader,num_parts=256'
        random_loader_dict = TrainingSummary.load_summary(TrainingSummary.output_folder, file_random_loader)
        precision_random = [float(x) for x in random_loader_dict['Precision']]

        file_graph_saint_random_walk_loader = 'Flickr: GraphSAINTRandomWalkSampler,walk_length:3,steps:12,batch:4096'
        graph_saint_random_walk_dict = TrainingSummary.load_summary(TrainingSummary.output_folder,
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

    @staticmethod
    def build(num_node_features: int, num_classes: int) -> GNNBaseModel:
        from torch_geometric.nn import GraphConv
        from dl.block.graph.gnn_base_block import GNNBaseBlock
        from dl.model.gnn_base_model import GNNBaseModel

        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gnn_block_1 = GNNBaseBlock(_id='K1',
                                   message_passing=conv_1,
                                   activation=nn.ReLU(),
                                   drop_out=0.2)
        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gnn_block_2 = GNNBaseBlock(_id='K2', message_passing=conv_2, activation=nn.ReLU(), drop_out=0.2)
        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gnn_block_3 = GNNBaseBlock(_id='K3', message_passing=conv_3, activation=nn.ReLU(), drop_out=0.2)

        ffnn_block = FFNNBlock.build(block_id='Output',
                                     layer=nn.Linear(3*hidden_channels, num_classes),
                                     activation=nn.LogSoftmax(dim=-1))
        return GNNBaseModel(model_id='Flickr',
                            gnn_blocks=[gnn_block_1, gnn_block_2, gnn_block_3],
                            ffnn_blocks=[ffnn_block])
