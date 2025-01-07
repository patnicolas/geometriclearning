import unittest

import os
from dl.block.graph.gnn_base_block import GNNBaseBlock
from dl.model.gnn_base_model import GNNBaseModel
from dl.training.exec_config import ExecConfig
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
        batch_size = 4
        walk_length = 6
        num_node_features = 24
        num_classes = 8
        gcn_model = GNNBaseModelTest.build(batch_size, walk_length, num_node_features, num_classes)
        print(repr(gcn_model))


    def test_train(self):
        from torch_geometric.datasets.flickr import Flickr

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
            encoding_len=101,  # No one-hot encoding
            weight_initialization=False)

        empty_cache: bool = False
        mix_precision: bool = False
        pin_memory: bool = False
        subset_size: int = 200

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size,
            monitor_memory=True,
            grad_accu_steps=1,
            device_config=None)

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _data = Flickr(path)
        _data.num_nodes = _data.num_node_features
        _data.num_edges = _data.num_edge_features
        gcn_model = GNNBaseModelTest.build(batch_size=4,
                                           walk_length=6,
                                           num_node_features = _data.num_nodes,
                                           num_classes = _data.num_classes)

        gcn_model.do_train(data=_data,
                           hyper_parameters=hyper_parameters,
                           metric_labels=['Precision', 'Recall'])

    @staticmethod
    def build(batch_size: int, walk_length: int, num_node_features: int, num_classes: int) -> GNNBaseModel:
        from torch_geometric.nn import BatchNorm

        hidden_channels = 256
        conv_1 = GraphConv(in_channels=num_node_features, out_channels=hidden_channels)
        gcn_conv_1 = GNNBaseBlock(_id='K1',
                                  message_passing=conv_1,
                                  activation=nn.ReLU(),
                                  batch_norm=BatchNorm(hidden_channels),
                                  drop_out=0.2)
        conv_2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
        gcn_conv_2 = GNNBaseBlock(_id='K2', message_passing=conv_2, activation=nn.ReLU())
        conv_3 = GraphConv(in_channels=hidden_channels, out_channels=num_classes)
        gcn_conv_3 = GNNBaseBlock(_id='K3', message_passing=conv_3)

        return GNNBaseModel.build(model_id='Karate club test',
                                  batch_size=batch_size,
                                  walk_length=walk_length,
                                  gcn_blocks=[gcn_conv_1, gcn_conv_2, gcn_conv_3])


