import unittest
import logging
import os
from typing import Dict, Any, AnyStr
import networkx as nx
import toponetx as tnx
from dataset.graph.pyg_datasets import PyGDatasets
from torch_geometric.data import Dataset
from topology.simplicial.graph_to_simplicial_complex import GraphToSimplicialComplex
from python import SKIP_REASON

TestErrorTypes = (ValueError, AssertionError, TypeError)

def lift_from_graph_cliques(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import graph_to_clique_complex

    logging.info('Graph lifted from NetworkX cliques with max rank 2')
    return graph_to_clique_complex(graph, max_rank=params.get('max_rank', 2))


def lift_from_graph_neighbors(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import graph_to_neighbor_complex

    logging.info('Graph lifted from node neighbors')
    return graph_to_neighbor_complex(graph)


def lift_from_graph_vietoris_rips(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import weighted_graph_to_vietoris_rips_complex
    r = 2
    max_dim = 3
    logging.info(f'Graph lifted from Vietoris Rips Complex with radius {r} and max_dim {max_dim}')
    return weighted_graph_to_vietoris_rips_complex(graph, r=r, max_dim=max_dim)


class GraphToSimplicialComplexTest(unittest.TestCase):

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_lift_cliques(self):
        try:
            dataset_name = 'CoraX'
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_cliques)
            logging.info(graph_to_simplicial.nx_graph)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 2708)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 5278)
        except TestErrorTypes as e:
            logging.error(e)
            self.assertFalse(False)

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_lift_neighbors(self):
        try:
            dataset_name = 'Cora'
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_neighbors)
            logging.info(graph_to_simplicial.nx_graph)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 2708)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 5278)
        except TestErrorTypes as e:
            logging.error(e)
            self.assertFalse(False)

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_lift_VR(self):
        try:
            pyg_dataset = PyGDatasets('Cora')
            graph_to_simplicial = GraphToSimplicialComplex[Dataset](dataset=pyg_dataset(),
                                                                    nx_graph=None,
                                                                    lifting_method=lift_from_graph_vietoris_rips)
            logging.info(graph_to_simplicial.nx_graph)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 2708)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 5278)
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_flickr_graph(self):
        try:
            dataset_name = "Flickr"
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_cliques)
            logging.info(graph_to_simplicial.nx_graph)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 89250)
            self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 449878)
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_faces_cliques(self):
        import time
        try:
            dataset_name = 'Cora'
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_cliques)
            start = time.time()
            tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})
            logging.info(f'Duration lift_from_graph_cliques for {dataset_name} {time.time() - start}')
            logging.info(tnx_simplicial)

            shape = tnx_simplicial.shape
            self.assertEqual(shape, (2708, 5278, 1630))
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_faces_neighbors(self):
        import time
        dataset_name = 'KarateClub'
        try:
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_neighbors)
            start = time.time()
            tnx_simplicial = graph_to_simplicial.add_faces()
            logging.info(f'Duration lift_from_graph_neighbors for {dataset_name} {time.time() - start}')
            logging.info(tnx_simplicial)
            shape = tnx_simplicial.shape

            self.assertEqual(len(shape), 18)
            self.assertEqual(shape[0], 34)
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_faces_vietoris_rips(self):
        dataset_name = 'KarateClub'
        try:
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_vietoris_rips)
            tnx_simplicial = graph_to_simplicial.add_faces()
            logging.info(tnx_simplicial)
            shape = tnx_simplicial.shape
            self.assertEqual(shape, (34, 78, 45, 11))
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed_count(self):
        import time
        try:
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset='PubMed',
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_cliques)
            start = time.time()
            tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})
            logging.info(f'Duration lift_from_graph_cliques {time.time() - start}')
            counts = GraphToSimplicialComplex.count_simplex_by_type(tnx_simplicial)
            logging.info(counts)
            self.assertEqual(counts['triangles'], 12520)
            self.assertEqual(counts['edges'], 44324)
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed_latency_neighbors(self):
        import time
        try:
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset='PubMed',
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_neighbors)
            start = time.time()
            tnx_simplicial = graph_to_simplicial.add_faces()
            logging.info(f'Duration lift_from_graph_neighbors {time.time() - start}')
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_features_from_hodge_laplacian(self):
        import time

        dataset_name = 'KarateClub'
        try:
            # Step 1: Configure the migration from Graph to Simplicial
            start = time.time()
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=dataset_name,
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_cliques)
            # Step 2: Add faces to existing graph nodes and edges
            tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})

            # Step 3: Generate the simplicial elements for nodes, edges and faces.
            #         Number of eigenvectors for node is 4, edges 5 and faces 4
            num_eigenvectors = (10, 9, 9)
            graph_complex_elements = GraphToSimplicialComplex.features_from_hodge_laplacian(tnx_simplicial, num_eigenvectors)
            logging.info(graph_complex_elements.dump(3))
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_features_from_hodge_laplacian_2(self):
        try:
            graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset='KarateClub',
                                                                   nx_graph=None,
                                                                   lifting_method=lift_from_graph_cliques)
            tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})
            num_eigenvectors = (4, 5, 4)
            graph_complex_elements =  GraphToSimplicialComplex.features_from_hodge_laplacian(tnx_simplicial,
                                                                                             num_eigenvectors)

            result = [face for idx, face, in enumerate(graph_complex_elements.featured_faces) if idx < 3]
            self.assertEqual(result[0].simplex_indices, (0, 1, 2))
            logging.info(result[0])
        except (ValueError, AssertionError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)
