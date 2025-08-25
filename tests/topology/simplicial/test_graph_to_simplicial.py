import unittest
import logging
import os
from dataset.graph.pyg_datasets import PyGDatasets
from topology.simplicial.graph_to_simplicial import GraphToSimplicial, SimplexType
from python import SKIP_REASON


class GraphToSimplicialTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_graph(self):
        simplicial_generator = GraphToSimplicial(dataset='Cora',
                                                 nx_graph=None,
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTetrahedrons)
        logging.info(simplicial_generator.nx_graph)
        self.assertEqual(simplicial_generator.nx_graph.number_of_nodes(), 2708)
        self.assertEqual(simplicial_generator.nx_graph.number_of_edges(), 5278)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_2_cora_graph(self):
        pyg_dataset = PyGDatasets('Cora')
        simplicial_generator = GraphToSimplicial(dataset=pyg_dataset(),
                                                 nx_graph=None,
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTetrahedrons)
        logging.info(simplicial_generator.nx_graph)
        self.assertEqual(simplicial_generator.nx_graph.number_of_nodes(), 2708)
        self.assertEqual(simplicial_generator.nx_graph.number_of_edges(), 5278)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_flickr_graph(self):
        simplicial_generator = GraphToSimplicial(dataset='Flickr',
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTetrahedrons)
        logging.info(simplicial_generator.nx_graph)
        self.assertEqual(simplicial_generator.nx_graph.number_of_nodes(), 89250)
        self.assertEqual(simplicial_generator.nx_graph.number_of_edges(), 449878)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_faces_features(self):
        simplicial_generator = GraphToSimplicial(dataset='KarateClub',
                                                 nx_graph=None,
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTetrahedrons)
        tnx_simplicial = simplicial_generator.add_faces()
        logging.info(tnx_simplicial)
        shape = tnx_simplicial.shape
        self.assertEqual(shape, (34, 78, 45, 11))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed_count(self):
        simplicial_generator = GraphToSimplicial(dataset='PubMed',
                                                 nx_graph=None,
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTriangles)
        tnx_simplicial = simplicial_generator.add_faces()
        counts = GraphToSimplicial.count_simplex_by_type(tnx_simplicial)
        logging.info(counts)
        self.assertEqual(counts['triangles'], 12520)
        self.assertEqual(counts['edges'], 44324)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_features_from_hodge_laplacian_pub_med(self):
        simplicial_generator = GraphToSimplicial(dataset='PubMed',
                                                 nx_graph=None,
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTriangles)
        tnx_simplicial = simplicial_generator.add_faces()
        num_eigenvectors = (4, 5, 4)
        node_simplicial_elements, edge_simplicial_elements, face_simplicial_elements = (
            GraphToSimplicial.features_from_hodge_laplacian(tnx_simplicial, num_eigenvectors)
        )
        result = [edge for idx, edge, in enumerate(edge_simplicial_elements) if idx < 3]
        self.assertEqual(result[0].node_indices, (0, 1378))
        logging.info(result[0])

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_features_from_hodge_laplacian_karate_club(self):
        simplicial_generator = GraphToSimplicial(dataset='KarateClub',
                                                 nx_graph=None,
                                                 max_num_nodes_cliques=21000,
                                                 simplex_types=SimplexType.WithTriangles)
        tnx_simplicial = simplicial_generator.add_faces()
        num_eigenvectors = (4, 5, 4)
        _, _, face_simplicial_elements = (
            GraphToSimplicial.features_from_hodge_laplacian(tnx_simplicial, num_eigenvectors)
        )
        result = [face for idx, face, in enumerate(face_simplicial_elements) if idx < 3]
        self.assertEqual(result[0].node_indices,  (0, 1, 2))
        logging.info(result[0])
