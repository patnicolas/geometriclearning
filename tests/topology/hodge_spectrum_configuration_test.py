import unittest
import logging
import python
from topology.hodge_spectrum_configuration import HodgeSpectrumConfiguration
from topology.simplicial import lift_from_graph_cliques
from topology.networkx_graph import NetworkxGraph

class HodgeSpectrumConfigurationTest(unittest.TestCase):

    def test_init_1(self):
        try:
            hodge_spectrum_config = HodgeSpectrumConfiguration(num_eigenvectors=(4, 5, 5))
            logging.info(hodge_spectrum_config)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_init_2(self):
        try:
            hodge_spectrum_config = HodgeSpectrumConfiguration(num_eigenvectors=(4, 0, 5))
            logging.info(hodge_spectrum_config)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(False)

    def test_init_3(self):
        try:
            hodge_spectrum_config = HodgeSpectrumConfiguration.build(num_node_eigenvectors=4,
                                                                     num_edge_eigenvectors=5,
                                                                     num_simplex_2_eigenvectors=5)
            logging.info(hodge_spectrum_config)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_generate_complex_1(self):
        dataset_name = 'Cora'
        try:
            networkx_graph = NetworkxGraph.build(dataset_name)
            hodge_spectrum_config = HodgeSpectrumConfiguration.build(num_node_eigenvectors=4,
                                                                     num_edge_eigenvectors=5,
                                                                     num_simplex_2_eigenvectors=5)
            simplicial_complex = lift_from_graph_cliques(graph=networkx_graph.G, params={'max_rank': 2})
            graph_complex_elements = hodge_spectrum_config.get_complex_features(simplicial_complex)
            logging.info(graph_complex_elements)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_generate_complex_2(self):
        dataset_name = 'PubMed'
        try:
            networkx_graph = NetworkxGraph.build(dataset_name)
            hodge_spectrum_config = HodgeSpectrumConfiguration.build(num_node_eigenvectors=4,
                                                                     num_edge_eigenvectors=5,
                                                                     num_simplex_2_eigenvectors=5)
            simplicial_complex = lift_from_graph_cliques(graph=networkx_graph.G, params={'max_rank': 2})
            graph_complex_elements = hodge_spectrum_config.get_complex_features(simplicial_complex)
            logging.info(graph_complex_elements)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)



