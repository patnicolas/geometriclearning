import unittest
from typing import AnyStr, List
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
import python


class ModelComparisonTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_train_models(self):
        dataset_name = 'Cora'
        hidden_channels = 256
        graph_SAGE_model = ModelComparisonTest.__create_GraphSAGE_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels)
        graph_SAGE_model.reset_parameters()
        logging.info(f'\nGraph SAGE model:\n{graph_SAGE_model}')
        graph_conv_model = ModelComparisonTest.__create_GraphConv_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels)
        graph_conv_model.reset_parameters()
        logging.info(f'\nGraph Conv model:\n{graph_conv_model}')
        train_data_loader, test_data_loader = ModelComparisonTest.__create_data_loaders(dataset_name=dataset_name,
                                                                                        neighbors=[8, 4])
        logging.info(f'\nTrain loader:\n{train_data_loader}')

        _dataset = ModelComparisonTest.__load_dataset('Cora')
        graph_data: torch_geometric.data.Data = _dataset[0]
        gnn_training = ModelComparisonTest.create_training_config(dataset_name=dataset_name,
                                                                  _data=graph_data,
                                                                  num_epochs=3,
                                                                  hidden_channels=hidden_channels)
        logging.info(f'\nGraph network trainer:\n{gnn_training}')
        model_comparison = ModelComparison(graph_SAGE_model, graph_conv_model)
        model_comparison.compare(gnn_training, train_data_loader, test_data_loader)
        self.assertTrue(True)

    def test_load_models(self):
        dataset_name = 'Cora'
        hidden_channels = 256
        graph_SAGE_model = ModelComparisonTest.__create_GraphSAGE_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels)
        graph_SAGE_model.reset_parameters()
        logging.info(f'\nGraph SAGE model:\n{graph_SAGE_model}')
        graph_conv_model = ModelComparisonTest.__create_GraphConv_model(dataset_name=dataset_name,
                                                                        hidden_channels=hidden_channels)

        model_comparison = ModelComparison(graph_SAGE_model, graph_conv_model)
        from plots.plotter import PlotterParameters
        plot_parameters_dict = {
            'count': 0,
            'x_label': 'Epochs',
            'y_label': 'Accuracy',
            'title': 'Accuracy plot',
            'fig_size': (10, 10)
        }
        plot_parameters = PlotterParameters.build(plot_parameters_dict)
        model_comparison.load(plot_parameters)
        self.assertTrue(True)


    @staticmethod
    def create_training_config(dataset_name: AnyStr, _data: Data, num_epochs: int, hidden_channels: int):
        class_weights = GraphDataLoader.class_weights(_data)
        training_attributes = {
            'dataset_name': dataset_name,
            # Model training Hyperparameters
            'learning_rate': 0.001,
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
                'batch_size': 32,
                'replace': True,
                'num_workers': 4
            },
            dataset_name=dataset_name
        )
        # 2. Extract the loader for training and validation sets
        return graph_data_loader()

    @staticmethod
    def __load_dataset(dataset_name: AnyStr) -> Dataset:
        pyg_dataset = PyGDatasets(dataset_name)
        return pyg_dataset()

    @staticmethod
    def __create_GraphConv_model(dataset_name: AnyStr, hidden_channels: int):
        _dataset = ModelComparisonTest.__load_dataset(dataset_name)
        _data: torch_geometric.data.Data = _dataset[0]

        graph_conv_block_1 = GraphConvBlock[GraphConv, None](
            block_id='Conv 24-256',
            graph_conv_layer=GraphConv(in_channels=_data.num_node_features, out_channels=hidden_channels),
            batch_norm_module=BatchNorm(hidden_channels),
            activation_module=nn.ReLU(),
            pooling_module=None,
            dropout_module=nn.Dropout(0.2)
        )
        graph_conv_block_2 = GraphConvBlock[GraphConv, None](
            block_id='Conv 256-256',
            graph_conv_layer=GraphConv(in_channels=hidden_channels, out_channels=hidden_channels),
            batch_norm_module=BatchNorm(hidden_channels),
            activation_module=nn.ReLU(),
            pooling_module=None,
            dropout_module=nn.Dropout(0.2)
        )
        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                             activation_module=nn.Softmax(dim=-1))

        return GraphConvModel[GraphConv, None](
            model_id=f'{dataset_name}_conv',
            graph_conv_blocks=[graph_conv_block_1, graph_conv_block_2],
            mlp_blocks=[mlp_block]
        )

    @staticmethod
    def __create_GraphSAGE_model(dataset_name: AnyStr, hidden_channels: int) -> GraphSAGEModel:
        _dataset = ModelComparisonTest.__load_dataset(dataset_name)
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

        mlp_block = MLPBlock(block_id='Fully connected',
                             layer_module=nn.Linear(hidden_channels, _dataset.num_classes),
                             activation_module=None)
        return GraphSAGEModel[SAGEConv](
            model_id=f'{dataset_name}_SAGE',
            graph_SAGE_blocks=[graph_SAGE_block_1, graph_SAGE_block_2],
            mlp_blocks=[mlp_block]
        )


if __name__ == '__main__':
    unittest.main()
