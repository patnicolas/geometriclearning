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
    def test_init_Flickr(self):
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]
        print(str(_data))
        self.assertTrue(len(_data.train_mask) > 1)

    @unittest.skip('Ignore')
    def test_init_movie_lens(self):
        from io import BytesIO
        import pandas as pd
        from urllib.request import urlopen
        from zipfile import ZipFile

        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        with urlopen(url) as zurl:
            with ZipFile(BytesIO(zurl.read())) as zfile:
                zfile.extractall('.')

        ratings = pd.read_csv('../ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
        print(f'Rating Movie lense\n{ratings}')



    @unittest.skip('Ignore')
    def test_call_1(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        import torch_geometric

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data: torch_geometric.data.data.Data = _dataset[0]

        graph_data_loader = GraphDataLoader(
            loader_attributes={'id': 'NeighborLoader', 'num_neighbors': [3, 2], 'batch_size': 4, 'replace': True},
            data=_data)
        print(f'Number of data points: {len(graph_data_loader)}')

        train_data_loader, test_data_loader = graph_data_loader()
        result = [f'{idx}: {str(batch)}' for idx, batch in enumerate(train_data_loader) if idx < 5]
        print('\n'.join(result))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()