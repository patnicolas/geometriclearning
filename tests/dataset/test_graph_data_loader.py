import unittest
from torch.utils.data import Dataset
from dataset.graph_data_loader import GraphDataLoader
from dataset import DatasetException


class GraphDataLoaderTest(unittest.TestCase):

    def test_init_1(self):
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]
        print(str(_data))
        self.assertTrue(len(_data.train_mask) > 1)

    def test_init_2(self):
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        try:
            graph_data_loader = GraphDataLoader(attributes_map = {'id':'NeighborLoader', 'batch_size': 4, 'replace': True},
                                                data= _dataset[0])
            print(str(graph_data_loader))
            self.assertTrue(False)
        except DatasetException as e:
            print(e)
            self.assertTrue(True)

    def test_init_3(self):
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        try:
            graph_data_loader = GraphDataLoader(
                attributes_map={'id': 'NeighborLoader', 'num_neighbors': 3, 'batch_size': 4, 'replace': True},
                data=_dataset[0])
            print(str(graph_data_loader))
            self.assertTrue(True)
        except DatasetException as e:
            print(e)
            self.assertTrue(False)

    def test_call(self):
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        try:
            graph_data_loader = GraphDataLoader(
                attributes_map={'id': 'NeighborLoader', 'num_neighbors': 3, 'batch_size': 4, 'replace': True},
                data=_dataset[0])
            train_data_loader, test_data_loader = graph_data_loader()
            self.assertTrue(True)
        except DatasetException as e:
            print(e)
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()