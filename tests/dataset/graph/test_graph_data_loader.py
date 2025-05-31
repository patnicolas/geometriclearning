import unittest
import logging

from dataset.graph.graph_data_loader import GraphDataLoader
from torch_geometric.data import Data
import torch

modules = ['torch', 'torch_sparse', 'torch_cluster', 'torch_scatter', 'torch_spline_conv']
from util import check_modules_availability
check_modules_availability(modules)


class GraphDataLoaderTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_graph_data(self):
        # Define the vertex-edge structure
        graph_descriptor = [[0, 0, 0, 0, 1],    # Source nodes/vertices
                            [1, 2, 3, 5, 4]]    # Target nodes/vertices
        edge_index_values = torch.tensor(data=graph_descriptor, dtype=torch.long)
        # Define the value or weight of each node
        node_values = torch.tensor(data=[1, 2, 3, 7, -1, -2], dtype=torch.float)
        graph_data = Data(x=node_values.T, edge_index=edge_index_values)

        logging.info(graph_data)
        # Request validation of the graph parameters
        self.assertTrue(graph_data.validate(raise_on_error=True))

    # @unittest.skip('Ignore')
    def test_random_node_flickr(self):
        dataset_name = 'Flickr'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'RandomNodeLoader',
                'num_parts': 256,
                'batch_size': 32,
                'num_workers': 2
            },
            dataset_name=dataset_name)
        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_random_node_flickr_2(self):
        dataset_name = 'Flickr'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'RandomNodeLoader',
                'num_parts': 64,
                'batch_size': 32,
                'num_workers': 2
            },
            dataset_name=dataset_name,
            num_subgraph_nodes=1024
         )
        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\nTrain data')
        logging.info('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_neighbor_node_flickr(self):
        dataset_name = 'Flickr'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': [4, 2],
                'replace': True,
                'batch_size': 8,
                'num_workers': 1
            },
            dataset_name=dataset_name,
            num_subgraph_nodes=64)

        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{id=}: {batch=}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_graph_SAINT_random_walk_cora(self):
        dataset_name = 'Cora'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'GraphSAINTRandomWalkSampler',
                'walk_length': 6,
                'sample_coverage': 64,
                'num_steps': 3,
                'batch_size': 4,
                'num_workers': 1
            },
            dataset_name=dataset_name)

        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_neighbor_node_facebook(self):
        dataset_name = 'Facebook'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': [5, 2],
                'replace': True,
                'batch_size': 8,
                'num_workers': 1
            },
            dataset_name=dataset_name,
            num_subgraph_nodes=64)

        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_graph_SAINT_node_karate_club(self):
        dataset_name = 'KarateClub'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'GraphSAINTNodeSampler',
                'sample_coverage': 32,
                'num_steps': 4,
                'batch_size': 128
            },
            dataset_name=dataset_name)

        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_cluster_proteins(self):
        dataset_name = 'PROTEINS'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
                'id': 'ClusterLoader',
                'num_parts': 128,
                'recursive': True,
                'batch_size': 128,
                'keep_inter_cluster_edges': True
            },
            dataset_name=dataset_name)
        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\n'.join(result))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
