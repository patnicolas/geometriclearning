__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch.utils.data import Dataset
from typing import AnyStr
from dataset import DatasetException

"""
PyTorch Geometric data sets varies in their definition and loading protocol. This class provides a clean, simple
functional interface to any PyTorch Geometric datasets.
Currently 20 data sets are supported
Examples:
        pyg_dataset = PyGDatasets('Cora')
        _dataset = pyg_dataset()
        data = _dataset[0]

"""

class PyGDatasets(object):
    def __init__(self, name: AnyStr) -> None:
        """
        Constructor for the interface to PyTorch Geometric dataset
        @param name: Name of the dataset
        @type name: AnyStr
        """
        self.name = name

    def __call__(self) -> Dataset:
        match self.name:
            case 'Cora' | 'PubMed' | 'CiteSeer':
                return self.__load_planetoid()
            case 'Facebook':
                return self.__load_facebook()
            case 'Flickr':
                return self.__load_flickr()
            case 'Wikipedia':
                return self.__load_wikipedia()
            case 'PROTEINS' | 'ENZYMES' | 'COLLAB' | 'REDDIT-BINARY':
                return self.__load_tu_dataset()
            case 'KarateClub':
                return self.__load_karate_club()
            case 'AmazonProducts':
                return self.__load_amazon_products()
            case 'Computers' | 'Photo':
                return self.__load_amazon()
            case 'Yelp':
                return self.__load_yelp()
            case 'HIV' | 'MUV' | 'PCBA' | 'ToxCast':
                return self.__load_molecule_net()
            case _:
                raise DatasetException(f'{self.name} data set is not supported')

    """ ------------------   Private helper methods --------------------- """
    def __load_molecule_net(self):
        from torch_geometric.datasets import MoleculeNet
        molecule_net = MoleculeNet(root='.', name=self.name)
        return molecule_net

    def __load_amazon(self):
        from torch_geometric.datasets import Amazon
        amazon_dataset = Amazon(root=',', name=self.name)
        return amazon_dataset

    def __load_yelp(self):
        from torch_geometric.datasets import Yelp
        yelp_dataset = Yelp('.')
        return yelp_dataset

    def __load_amazon_products(self) -> Dataset:
        from torch_geometric.datasets import AmazonProducts
        amazon_products_dataset = AmazonProducts(root='.')
        return amazon_products_dataset

    def __load_karate_club(self) -> Dataset:
        from torch_geometric.datasets import KarateClub
        return KarateClub()

    def __load_tu_dataset(self) -> Dataset:
        from torch_geometric.datasets import TUDataset
        tmp_dir = f'/tmp/{self.name}'
        dataset = TUDataset(root=tmp_dir, name=self.name)
        return dataset.shuffle()

    def __load_wikipedia(self) -> Dataset:
        from torch_geometric.datasets import WikipediaNetwork
        import torch_geometric.transforms as T
        _dataset = WikipediaNetwork(root=".", name="chameleon", transform=T.RandomNodeSplit(num_val=200, num_test=500))
        return _dataset

    def __load_flickr(self) -> Dataset:
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        return _dataset

    def __load_planetoid(self) -> Dataset:
        from torch_geometric.datasets import Planetoid
        _dataset = Planetoid(root=".", name=self.name)
        return _dataset

    def __load_facebook(self) -> Dataset:
        from torch_geometric.datasets import FacebookPagePage
        _dataset: Dataset = FacebookPagePage(root=".")
        return _dataset
