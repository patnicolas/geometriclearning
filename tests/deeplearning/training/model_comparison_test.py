import unittest
from typing import AnyStr, List, Tuple
from deeplearning.model.graph.graph_sage_model import GraphSAGEModel
import torch.nn as nn
from deeplearning.block.graph.graph_sage_block import GraphSAGEBlock
import torch_geometric
from dataset.graph.graph_data_loader import GraphDataLoader
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.block.graph.graph_conv_block import GraphConvBlock
from deeplearning.training.gnn_training import GNNTraining
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.graph.graph_conv_model import GraphConvModel
from torch_geometric.nn import SAGEConv, GraphConv, BatchNorm
from torch_geometric.data import Data, Dataset
from deeplearning.training.model_comparison import ModelComparison
import logging
from plots.plotter import PlotterParameters
import python


class ModelComparisonTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_train_models(self):
        # Configurable Parameters
        dataset_name = 'Cora'
        hidden_channels = 32
        neighbors = [8, 4]
        num_blocks = (4, 2)
        lr = 0.0012
        num_epochs = 6
        graph_SAGE_model = ModelComparisonTest.__create_GraphSAGE_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels,
                                                                        num_blocks=num_blocks[0],
                                                                        desc=f'NeighborLoader{neighbors}_{lr}')
        graph_SAGE_model.reset_parameters()
        logging.info(f'\nGraph SAGE model:\n{graph_SAGE_model}')
        graph_conv_model = ModelComparisonTest.__create_GraphConv_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels,
                                                                        num_blocks=num_blocks[1],
                                                                        desc=f'NeighborLoader{neighbors}_{lr}')
        graph_conv_model.reset_parameters()
        logging.info(f'\nGraph Conv model:\n{graph_conv_model}')
        train_data_loader, test_data_loader = ModelComparisonTest.__create_data_loaders(dataset_name=dataset_name,
                                                                                        neighbors=neighbors)
        logging.info(f'\nTrain loader:\n{train_data_loader}')

        _dataset = ModelComparisonTest.__load_dataset(dataset_name)
        graph_data: torch_geometric.data.Data = _dataset[0]
        gnn_training = ModelComparisonTest.__create_training_config(dataset_name=dataset_name,
                                                                    _data=graph_data,
                                                                    num_epochs=num_epochs,
                                                                    hidden_channels=hidden_channels,
                                                                    lr=lr)
        logging.info(f'\nGraph network trainer:\n{gnn_training}')
        # model_comparison = ModelComparison([graph_SAGE_model, graph_conv_model])
        model_comparison = ModelComparison([graph_SAGE_model, graph_conv_model])
        model_comparison.compare(gnn_training, train_data_loader, test_data_loader)

        ModelComparisonTest.__load_and_plot(dataset_name, hidden_channels, neighbors, num_blocks, lr)
        self.assertTrue(True)

    # @unittest.skip('Ignore')
    def test_load_and_plot(self):
        # Configurable Parameters
        dataset_name = 'Cora'
        neighbors = [20, 12]
        num_blocks = (2, 4)
        ModelComparisonTest.__plot(dataset_name, neighbors, num_blocks)

    @staticmethod
    def __plot(dataset_name: AnyStr, neighbors: List[int], num_blocks: Tuple[int, int]):
        from plots.plotter import Plotter
        from metric.performance_metrics import PerformanceMetrics

        plot_parameters_dict = {
            'count': 0,
            'x_label': 'Epochs',
            'y_label': 'Accuracy',
            'title': 'Accuracy plot',
            'fig_size': (8, 8),
            'multi_plot_pause': 0
        }
        plot_parameters = PlotterParameters.build(plot_parameters_dict)
        model_ids = [
            f'GraphSAGE_{dataset_name}_Neighbors{neighbors}_{num_blocks[0]}layers',
            f'GraphConv_{dataset_name}_Neighbors{neighbors}_{num_blocks[0]}layers',
            f'GraphSAGE_{dataset_name}_Neighbors{neighbors}_{num_blocks[1]}layers',
            f'GraphConv_{dataset_name}_Neighbors{neighbors}_{num_blocks[1]}layers'
        ]

        perf_metric_models = [PerformanceMetrics.load_summary(model_id) for model_id in model_ids]

        # Merge performance metrics across multiple models
        merged_metrics = {k: [perf_metric_model[k] for perf_metric_model in perf_metric_models]
                          for k in perf_metric_models[0].keys()}

        # Setup plotting parameters then display the plot
        for k, values in merged_metrics.items():
            setattr(plot_parameters, 'y_label', k)
            setattr(plot_parameters, 'title', f'{k} Comparison')
            Plotter.plot(values, model_ids, plot_parameters)

    @staticmethod
    def __load_and_plot(dataset_name: AnyStr,
                        hidden_channels: int,
                        neighbors: List[int],
                        num_blocks: Tuple[int, int],
                        lr: float):
        graph_SAGE_model = ModelComparisonTest.__create_GraphSAGE_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels,
                                                                        num_blocks=num_blocks[0],
                                                                        desc=f'Neighbors{neighbors}_{lr}')
        graph_SAGE_model.reset_parameters()
        logging.info(f'\nGraph SAGE model:\n{graph_SAGE_model}')
        graph_conv_model = ModelComparisonTest.__create_GraphConv_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels,
                                                                        num_blocks=num_blocks[1],
                                                                        desc=f'Neighbors{neighbors}_{lr}')

        model_comparison = ModelComparison([graph_SAGE_model, graph_conv_model])
        from plots.plotter import PlotterParameters
        plot_parameters_dict = {
            'count': 0,
            'x_label': 'Epochs',
            'y_label': 'Accuracy',
            'title': 'Accuracy plot',
            'fig_size': (8, 8),
            'multi_plot_pause': 0
        }
        plot_parameters = PlotterParameters.build(plot_parameters_dict)
        model_comparison.load_and_plot(plot_parameters)

    @staticmethod
    def __create_training_config(dataset_name: AnyStr, _data: Data, num_epochs: int, hidden_channels: int, lr: float):
        class_weights = GraphDataLoader.class_weights(_data)
        training_attributes = {
            'dataset_name': dataset_name,
            # Model training Hyperparameters
            'learning_rate': lr,
            'batch_size': 32,
            'loss_function': nn.CrossEntropyLoss(label_smoothing=0.05, reduction='mean'),
            'momentum': 0.95,
            'weight_decay': 2e-3,
            'encoding_len': -1,
            'train_eval_ratio': 0.9,
            'weight_initialization': 'kaiming',
            'optim_label': 'adam',
            'drop_out': 0.25,
            'is_class_imbalance': True,
            'class_weights': class_weights,
            'patience': 2,
            'min_diff_loss': 0.02,
            'epochs': num_epochs,
            # Model configuration
            'hidden_channels': hidden_channels,
            # Performance metric definition
            'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR']
        }
        return GNNTraining.build(training_attributes)

    @staticmethod
    def __create_data_loaders(dataset_name: AnyStr, neighbors: List[int]) -> (GraphDataLoader, GraphDataLoader):
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': neighbors,
                'batch_size': 64,
                'replace': True,
                'num_workers': 4
            },
            dataset_name=dataset_name,
            num_subgraph_nodes=0
        )
        # 2. Extract the loader for training and validation sets
        return graph_data_loader()

    @staticmethod
    def __load_dataset(dataset_name: AnyStr) -> Dataset:
        pyg_dataset = PyGDatasets(dataset_name)
        return pyg_dataset()

    @staticmethod
    def __create_GraphConv_model(dataset_name: AnyStr,
                                 hidden_channels: int,
                                 num_blocks: int,
                                 desc: AnyStr) -> GraphConvModel:
        _dataset = ModelComparisonTest.__load_dataset(dataset_name)
        _data: torch_geometric.data.Data = _dataset[0]

        graph_conv_blocks = []
        graph_conv_block_1 = GraphConvBlock[GraphConv, None](
            block_id=f'Conv {_data.num_node_features}-{hidden_channels}',
            graph_conv_layer=GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels),
            batch_norm_module=None,
            activation_module=nn.ReLU(),
            pooling_module=None,
            dropout_module=nn.Dropout(0.25)
        )
        graph_conv_blocks.append(graph_conv_block_1)
        graph_conv_block_2 = GraphConvBlock[GraphConv, None](
            block_id=f'Conv {hidden_channels}-{hidden_channels}',
            graph_conv_layer=GraphConv(in_channels=hidden_channels, out_channels=hidden_channels),
            batch_norm_module=None,
            activation_module=nn.ReLU(),
            pooling_module=None,
            dropout_module=nn.Dropout(0.25)
        )
        graph_conv_blocks.append(graph_conv_block_2)
        if num_blocks > 2:
            for _ in range(num_blocks):
                graph_conv_block = GraphConvBlock[GraphConv, None](
                    block_id=f'Conv {hidden_channels}-{hidden_channels}',
                    graph_conv_layer=GraphConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    batch_norm_module=None,
                    activation_module=nn.ReLU(),
                    pooling_module=None,
                    dropout_module=nn.Dropout(0.25)
                )
                graph_conv_blocks.append(graph_conv_block)

        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                             activation_module=None)

        return GraphConvModel[GraphConv, None](
            model_id=f'{dataset_name}_conv_{num_blocks}blocks_{desc}',
            graph_conv_blocks=graph_conv_blocks,
            mlp_blocks=[mlp_block]
        )

    @staticmethod
    def __create_GraphSAGE_model(dataset_name: AnyStr,
                                 hidden_channels: int,
                                 num_blocks: int,
                                 desc: AnyStr) -> GraphSAGEModel:
        # Load the appropriate dataset
        _dataset = ModelComparisonTest.__load_dataset(dataset_name)
        _data: torch_geometric.data.Data = _dataset[0]

        graph_SAGE_blocks = []
        # First GraphSAGE convolutional layer
        sage_conv_1 = SAGEConv(in_channels=_data.num_node_features, out_channels=hidden_channels, aggr='mean')
        graph_SAGE_block_1 = GraphSAGEBlock[SAGEConv](block_id=f'SAGE {_data.num_node_features}-{hidden_channels}',
                                                      graph_SAGE_layer=sage_conv_1,
                                                      batch_norm_module=None, #BatchNorm(hidden_channels),
                                                      activation_module=nn.ReLU(),
                                                      dropout_module=nn.Dropout(0.2))
        graph_SAGE_blocks.append(graph_SAGE_block_1)

        # Second GraphSAGE convolutional layer
        sage_conv_2 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels, aggr='mean')
        graph_SAGE_block_2 = GraphSAGEBlock[SAGEConv](block_id=f'SAGE {hidden_channels}-{hidden_channels}',
                                                      graph_SAGE_layer=sage_conv_2,
                                                      batch_norm_module=None,
                                                      activation_module=nn.ReLU(),
                                                      dropout_module=nn.Dropout(0.25))
        graph_SAGE_blocks.append(graph_SAGE_block_2)
        # If this model contains more than 2 graph neural blocks...
        if num_blocks > 2:
            for _ in range(num_blocks):
                sage_conv_3 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels, aggr='mean')
                graph_SAGE_block_3 = GraphSAGEBlock[SAGEConv](block_id=f'SAGE {hidden_channels}-{hidden_channels}',
                                                              graph_SAGE_layer=sage_conv_3,
                                                              batch_norm_module=None,
                                                              activation_module=nn.ReLU(),
                                                              dropout_module=nn.Dropout(0.25))
                graph_SAGE_blocks.append(graph_SAGE_block_3)

        # The output block is a linear fully-connected block
        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                             activation_module=None)
        return GraphSAGEModel[SAGEConv](
            model_id=f'{dataset_name}_SAGE_{num_blocks}blocks_{desc}',
            graph_SAGE_blocks=graph_SAGE_blocks,
            mlp_blocks=[mlp_block]
        )


if __name__ == '__main__':
    unittest.main()
