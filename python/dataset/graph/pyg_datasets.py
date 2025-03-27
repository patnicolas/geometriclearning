__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch.utils.data import Dataset
from typing import AnyStr, Optional
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
    dataset_dict = {
        'Cora': lambda pyg: pyg.__load_planetoid(),
        'PubMed': lambda pyg: pyg.__load_planetoid(),
        'CiteSeer': lambda pyg: pyg.__load_planetoid(),
        'Facebook': lambda pyg: pyg.__load_facebook(),
        'Flickr': lambda pyg: pyg.__load_flickr(),
        'Wikipedia': lambda pyg: pyg.__load_wikipedia(),
        'PROTEINS': lambda pyg: pyg.__load_tu_dataset(),
        'ENZYMES': lambda pyg: pyg.__load_tu_dataset(),
        'COLLAB': lambda pyg: pyg.__load_tu_dataset(),
        'REDDIT-BINARY': lambda pyg: pyg.__load_tu_dataset(),
        'KarateClub': lambda pyg: pyg.__load_karate_club(),
        'AmazonProducts': lambda pyg: pyg.__load_amazon_products(),
        'Computers': lambda pyg: pyg.__load_amazon(),
        'Photo': lambda pyg: pyg.__load_amazon(),
        'Yelp': lambda pyg: pyg.__load_yelp(),
        'HIV': lambda pyg: pyg.__load_molecule_net(),
        'MUV': lambda pyg: pyg.__load_molecule_net(),
        'PCBA': lambda pyg: pyg.__load_molecule_net(),
        'ToxCast': lambda pyg: pyg.__load_molecule_net(),
    }

    def __init__(self, name: AnyStr) -> None:
        """
        Constructor for the interface to PyTorch Geometric dataset
        @param name: Name of the dataset
        @type name: AnyStr
        """
        self.name = name

    def __call__(self) -> Optional[Dataset]:
        """
        Method to load and extract data set from any supported source
        @return data set from PyTorch Geometric library
        @ttype torch.util.Dataset
        """
        try:
            func = PyGDatasets.dataset_dict[self.name]
            return func(self)
        except KeyError as err:
            raise DatasetException(f'Dataset {self.name} not supported {err}')

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
