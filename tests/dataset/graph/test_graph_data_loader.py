import unittest
from torch.utils.data import Dataset
from dataset.graph.graph_data_loader import GraphDataLoader
from torch_geometric.data import Data
import torch


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

        print(graph_data)
        # Request validation of the graph parameters
        self.assertTrue(graph_data.validate(raise_on_error=True))

    @unittest.skip('Ignore')
    def test_random_node_flickr(self):
        dataset_name = 'Flickr'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            loader_attributes={
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
        print('\n'.join(result))
        self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_neighbor_node_flickr(self):
        dataset_name = 'Flickr'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            loader_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': [4, 2],
                'replace': True,
                'batch_size': 64,
                'num_workers': 4
            },
            dataset_name=dataset_name)

        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        print('\n'.join(result))
        self.assertTrue(True)


    def test_neighbor_node_cora(self):
        dataset_name = 'Cora'
        # 1. Initialize the loader
        graph_data_loader = GraphDataLoader(
            loader_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': [5, 2],
                'replace': True,
                'batch_size': 128,
                'num_workers': 1
            },
            dataset_name=dataset_name)

        # 2. Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        print('\n'.join(result))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
