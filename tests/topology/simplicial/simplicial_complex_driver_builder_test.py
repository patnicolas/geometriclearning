import unittest
import logging
import python
from topology.simplicial.simplicial_complex_driver_builder import SimplicialComplexDriverBuilder
from topology.simplicial import lift_from_graph_cliques


class SimplicialComplexDriverBuilderTest(unittest.TestCase):

    def test_initialization(self):
        dataset = 'Cora'
        simplicial_complex_builder = SimplicialComplexDriverBuilder(dataset=dataset, nx_graph=None)
        logging.info(simplicial_complex_builder)
        self.assertTrue(True)

    def test_generation(self):
        dataset = 'Cora'
        simplicial_complex_builder = SimplicialComplexDriverBuilder(dataset=dataset, nx_graph=None)
        simplicial_complex = simplicial_complex_builder(num_eigenvectors=(4, 3, 4),
                                                        lifting_method=lift_from_graph_cliques)
        logging.info(simplicial_complex)
        self.assertTrue(simplicial_complex is not None)