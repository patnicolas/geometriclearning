import unittest

import os
from dl.block.graph.g_message_passing_block import GMessagePassingBlock
from dl.model.gnn_base_model import GNNBaseModel
from dl.training.hyper_params import HyperParams
from torch_geometric.nn import GraphConv
import torch.nn as nn


class GNNBaseModelTest(unittest.TestCase):

    def test_init_1(self):
        import numpy as np
        import torch
        dense_array = np.array([
            [0, 0, 3],
            [4, 0, 0],
            [0, 0, 0]
        ])

        # Step 2: Convert to COO (coordinate) format
        rows, cols = np.nonzero(dense_array)  # Get the indices of non-zero elements
        values = dense_array[rows, cols]  # Get the values of non-zero elements

        # Step 3: Create a PyTorch sparse tensor
        indices = torch.tensor([rows, cols], dtype=torch.int64)  # Indices of non-zero elements
        values = torch.tensor(values, dtype=torch.float32)  # Corresponding values
        size = dense_array.shape  # Shape of the original array

        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)

        # Verify the sparse tensor
        print("Sparse Tensor:")
        print(sparse_tensor)

    def test_init_2(self):
        num_node_features = 24
        num_classes = 8
        gcn_model = GNNBaseModelTest.build(num_node_features, num_classes)
        print(repr(gcn_model))

    def test_train(self):
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        lr = 0.001
        hyper_parameters = HyperParams(
            lr=lr,
            momentum=0.89,
            epochs=2,
            optim_label='adam',
            batch_size=4,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            encoding_len=101)

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        num_nodes = _dataset[0].num_node_features
        gcn_model = GNNBaseModelTest.build(num_node_features=num_nodes,
                                           num_classes=_dataset[0].num_classes)

        gcn_model.do_train(hyper_parameters=hyper_parameters,
                           metric_labels=['Precision', 'Recall'])

    @staticmethod
    def build(num_node_features: int, num_classes: int) -> GNNBaseModel:
        from torch_geometric.nn import BatchNorm

        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gcn_conv_1 = GMessagePassingBlock(block_id='K1',
                                          message_passing_module=conv_1,
                                          activation_module=nn.ReLU(),
                                          batch_norm_module=BatchNorm(hidden_channels),
                                          drop_out_module=0.2)
        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gcn_conv_2 = GMessagePassingBlock(block_id='K2', message_passing_module=conv_2, activation_module=nn.ReLU())
        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=num_classes)
        gcn_conv_3 = GMessagePassingBlock(block_id='K3', message_passing_module=conv_3)

        return GNNBaseModel.build(model_id='Karate club test',
                                  gnn_blocks=[gcn_conv_1, gcn_conv_2, gcn_conv_3])


