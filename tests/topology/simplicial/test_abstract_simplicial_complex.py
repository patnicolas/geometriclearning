import unittest
import numpy as np
from topology.simplicial.abstract_simplicial_complex import AbstractSimplicialComplex, SimplicialElement
from topology.simplicial.simplicial_laplacian import SimplicialLaplacian, SimplicialLaplacianType
import logging
import python


class AbstractSimplicialComplexTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_simplicial_node(self):
        simplicial_node = SimplicialElement(feature_set=np.array([0.45, 0.76, 0.05]))
        entry = simplicial_node((4,))
        logging.info(entry)
        self.assertEqual(entry[0], (4,))

    @unittest.skip('Ignore')
    def test_simplicial_edge(self):
        simplicial_edge = SimplicialElement(node_indices=(4, 9), feature_set=np.array([0.09, 0.61, 0.50]))
        entry = simplicial_edge()
        logging.info(entry)
        self.assertEqual(entry[0], (4, 9))

    @unittest.skip('Ignore')
    def test_simplicial_face(self):
        simplicial_face = SimplicialElement(node_indices=(4, 9, 8), feature_set=np.array([0.17, 0.22, 0.1, 0.99]))
        entry = simplicial_face()
        logging.info(entry)
        self.assertEqual(entry[0], (4, 9, 8))

    @unittest.skip('Ignore')
    def test_simplicial_maps(self):
        simplicial_nodes = [
            SimplicialElement(feature_set=np.array([0.45, 0.76, 0.05])),
            SimplicialElement(feature_set=np.array([0.58, 0.01, 0.59])),
            SimplicialElement(feature_set=np.array([0.33, 0.80, 0.00]))
            ]
        simplicial_edges = [
            SimplicialElement(node_indices=(1, 2), feature_set=np.array([0.09, 0.61])),
            SimplicialElement(node_indices=(2, 3), feature_set=np.array([0.00, 0.87])),
        ]
        simplicial_faces = [SimplicialElement(node_indices=(1, 2, 3), feature_set=np.array([0.17, 0.22, 0.1, 0.99]))]
        abstract_simplicial_complex = AbstractSimplicialComplex(simplicial_nodes, simplicial_edges, simplicial_faces)
        node_map = abstract_simplicial_complex.node_features_map()
        edge_map = abstract_simplicial_complex.edge_features_map()
        face_map = abstract_simplicial_complex.face_features_map()
        logging.info(f'\nNodes map:\n{node_map}\nEdges map:\n{edge_map}\nFaces map:\n{face_map}')

    @unittest.skip('Ignore')
    def test_init_1(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=4,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            logging.info(str(abstract_simplicial_complex))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_2(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=3,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            logging.info(str(abstract_simplicial_complex))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_simplicial_up_laplacian_compute(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=5,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            # simplicial_feature_set.show()
            simplicial_laplacian_0 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                         rank=0,
                                                         signed=True)
            up_laplacian_rk0 = abstract_simplicial_complex.laplacian(simplicial_laplacian_0)
            logging.info(f'\nUP-Laplacian rank 0\n{up_laplacian_rk0}')
            simplicial_laplacian_1 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                         rank=1,
                                                         signed=True)
            up_laplacian_rk1 = abstract_simplicial_complex.laplacian(simplicial_laplacian_1)
            logging.info(f'\nUP-Laplacian rank 1\n{up_laplacian_rk1}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_simplicial_down_laplacian_compute(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=4,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            simplicial_laplacian_1 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                rank=1,
                signed=True)
            down_laplacian_rk1 = abstract_simplicial_complex.laplacian(simplicial_laplacian_1)
            logging.info(f'\nDown-Laplacian rank 1\n{down_laplacian_rk1}')
            simplicial_laplacian_2 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                rank=2,
                signed=True)
            down_laplacian_rk2 = abstract_simplicial_complex.laplacian(simplicial_laplacian_2)
            logging.info(f'\nDown-Laplacian rank 2\n{down_laplacian_rk2}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_simplicial_hodge_laplacian_compute(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=5,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            simplicial_laplacian_0 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=0,
                signed=True)
            hodge_laplacian_rk0 = abstract_simplicial_complex.laplacian(simplicial_laplacian_0)
            logging.info(f'\nHodge-Laplacian rank 0\n{hodge_laplacian_rk0}')

            simplicial_laplacian_1 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=1,
                signed=True)
            hodge_laplacian_rk1 = abstract_simplicial_complex.laplacian(simplicial_laplacian_1)
            logging.info(f'\nHodge-Laplacian rank 1\n{hodge_laplacian_rk1}')

            simplicial_laplacian_2 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=2,
                signed=True
            )
            hodge_laplacian_rk2 = abstract_simplicial_complex.laplacian(simplicial_laplacian_2)
            logging.info(f'\nHodge-Laplacian rank 2\n{hodge_laplacian_rk2}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_adjacency(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=4,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            logging.info(f'\nAdjacency matrix:\n{abstract_simplicial_complex.adjacency_matrix()}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_directed_1(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=5,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            for rank in range(0, 3):
                incidence_matrix = abstract_simplicial_complex.incidence_matrix(rank=rank)
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{incidence_matrix}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_directed_2(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            abstract_simplicial_complex = AbstractSimplicialComplex.random(node_feature_dimension=5,
                                                                           edge_node_indices=edge_set,
                                                                           face_node_indices=face_set)
            for rank in range(0, 3):
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{abstract_simplicial_complex.incidence_matrix(rank=rank)}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)



