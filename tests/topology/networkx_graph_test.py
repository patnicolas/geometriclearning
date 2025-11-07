import unittest
import logging
import python
from topology.networkx_graph import NetworkxGraph

class NetworkxGraphTests(unittest.TestCase):

    def test_init_1(self):
        try:
            dataset_name = 'PubMed'

            from dataset.graph.pyg_datasets import PyGDatasets

            # The class PyGDatasets validate the dataset is part of PyTorch Geometric Library
            pyg_dataset = PyGDatasets(dataset_name)
            dataset = pyg_dataset()

            networkx_graph = NetworkxGraph(dataset[0])
            self.assertTrue(networkx_graph.G is not None)
            logging.info(networkx_graph)
        except TypeError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_init_2(self):
        try:
            dataset_name = 'PubMed'
            networkx_graph = NetworkxGraph.build(dataset_name)
            self.assertTrue(networkx_graph.G is not None)
            logging.info(networkx_graph)
        except TypeError as e:
            logging.error(e)
            self.assertFalse(True)