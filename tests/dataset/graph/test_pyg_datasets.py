import unittest
from dataset.graph.pyg_datasets import PyGDatasets
from typing import AnyStr

class PyGDatasetsTest(unittest.TestCase):

    def test_cite_seer(self):
        PyGDatasetsTest.__load_show_data('CiteSeer')
        self.assertTrue(True)

    def test_molecule_net(self):
        PyGDatasetsTest.__load_show_data('MUV')
        PyGDatasetsTest.__load_show_data('HIV')
        self.assertTrue(True)


    def test_amazon_computers(self):
        PyGDatasetsTest.__load_show_data('Computers')
        self.assertTrue(True)
    
    def test_yelp(self):
        PyGDatasetsTest.__load_show_data('Yelp')
        self.assertTrue(True)
    
    def test_amazon_products(self):
        PyGDatasetsTest.__load_show_data('AmazonProducts')

    def test_init_planetoid(self):
        PyGDatasetsTest.__load_show_data('Cora')
        PyGDatasetsTest.__load_show_data('PubMed')
        PyGDatasetsTest.__load_show_data('CiteSeer')
        self.assertTrue(True)

    def test_init_tu_dataset(self):
        PyGDatasetsTest.__load_show_data('PROTEINS')
        PyGDatasetsTest.__load_show_data('ENZYMES')
        PyGDatasetsTest.__load_show_data('COLLAB')
        self.assertTrue(True)

    def test_init_facebook(self):
        PyGDatasetsTest.__load_show_data('Facebook')
        self.assertTrue(True)

    def test_init_wikipedia(self):
        PyGDatasetsTest.__load_show_data('Wikipedia')
        self.assertTrue(True)

    def test_Karate_club(self):
        PyGDatasetsTest.__load_show_data('KarateClub')
        self.assertTrue(True)

    @staticmethod
    def __load_show_data(name: AnyStr) -> None:
        pyg_dataset = PyGDatasets(name)
        _dataset = pyg_dataset()
        data = _dataset[0]
        print(f'\n{name}:\n{data}')