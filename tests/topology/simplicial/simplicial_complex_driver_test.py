import unittest
import numpy as np

from topology.graph_complex_elements import GraphComplexElements
from topology.simplicial.simplicial_complex_driver import SimplicialComplexDriver
from topology.complex_element import ComplexElement
from topology.simplicial.simplicial_laplacian import SimplicialLaplacian, SimplicialLaplacianType
import logging
import python


class SimplicialComplexDriverTest(unittest.TestCase):

    def test_simplicial_node(self):
        try:
            simplicial_node = ComplexElement(feature_set=np.array([0.45, 0.76, 0.05]))
            entry = simplicial_node((4,))
            logging.info(entry)
            self.assertEqual(entry[0], (4,))
        except (ValueError, AssertionError) as e:
            logging.error(e)
            self.assertFalse(True)

    def test_simplicial_edge(self):
        try:
            simplicial_edge = ComplexElement(node_indices=(4, 9), feature_set=np.array([0.09, 0.61, 0.50]))
            entry = simplicial_edge()
            logging.info(entry)
            self.assertEqual(entry[0], (4, 9))
        except (ValueError, AssertionError) as e:
            logging.error(e)
            self.assertFalse(True)

    def test_simplicial_face(self):
        try:
            simplicial_face = ComplexElement(node_indices=(4, 9, 8), feature_set=np.array([0.17, 0.22, 0.1, 0.99]))
            entry = simplicial_face()
            logging.info(entry)
            self.assertEqual(entry[0], (4, 9, 8))
        except (ValueError, AssertionError, ValueError) as e:
            logging.error(e)
            self.assertFalse(True)

    def test_simplicial_maps(self):
        simplicial_nodes = [
            ComplexElement(feature_set=np.array([0.45, 0.76, 0.05])),
            ComplexElement(feature_set=np.array([0.58, 0.01, 0.59])),
            ComplexElement(feature_set=np.array([0.33, 0.80, 0.00]))
            ]
        simplicial_edges = [
            ComplexElement(node_indices=(1, 2), feature_set=np.array([0.09, 0.61])),
            ComplexElement(node_indices=(2, 3), feature_set=np.array([0.00, 0.87])),
        ]
        simplicial_faces = [ComplexElement(node_indices=(1, 2, 3), feature_set=np.array([0.17, 0.22, 0.1, 0.99]))]
        graph_complex_elements = GraphComplexElements(simplicial_nodes, simplicial_edges, simplicial_faces)

        simplicial_complex_driver = SimplicialComplexDriver(graph_complex_elements)
        node_map = simplicial_complex_driver.node_features_map()
        edge_map = simplicial_complex_driver.edge_features_map()
        face_map = simplicial_complex_driver.face_features_map()
        logging.info(f'\nNodes map:\n{node_map}\nEdges map:\n{edge_map}\nFaces map:\n{face_map}')

    def test_init_1(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=4,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            logging.info(str(simplicial_complex_driver))
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_2(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=3,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            logging.info(str(simplicial_complex_driver))
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_simplicial_up_laplacian_compute(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=5,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            # simplicial_feature_set.show()
            simplicial_laplacian_0 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                         rank=0,
                                                         signed=True)
            up_laplacian_rk0 = simplicial_complex_driver.laplacian(simplicial_laplacian_0)
            logging.info(f'\nUP-Laplacian rank 0\n{up_laplacian_rk0}')
            simplicial_laplacian_1 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                         rank=1,
                                                         signed=True)
            up_laplacian_rk1 = simplicial_complex_driver.laplacian(simplicial_laplacian_1)
            logging.info(f'\nUP-Laplacian rank 1\n{up_laplacian_rk1}')
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_simplicial_down_laplacian_compute(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=4,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            simplicial_laplacian_1 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                rank=1,
                signed=True)
            down_laplacian_rk1 = simplicial_complex_driver.laplacian(simplicial_laplacian_1)
            logging.info(f'\nDown-Laplacian rank 1\n{down_laplacian_rk1}')
            simplicial_laplacian_2 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                rank=2,
                signed=True)
            down_laplacian_rk2 = simplicial_complex_driver.laplacian(simplicial_laplacian_2)
            logging.info(f'\nDown-Laplacian rank 2\n{down_laplacian_rk2}')
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_simplicial_hodge_laplacian_compute(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=5,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            simplicial_laplacian_0 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=0,
                signed=True)
            hodge_laplacian_rk0 = simplicial_complex_driver.laplacian(simplicial_laplacian_0)
            logging.info(f'\nHodge-Laplacian rank 0\n{hodge_laplacian_rk0}')

            simplicial_laplacian_1 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=1,
                signed=True)
            hodge_laplacian_rk1 = simplicial_complex_driver.laplacian(simplicial_laplacian_1)
            logging.info(f'\nHodge-Laplacian rank 1\n{hodge_laplacian_rk1}')

            simplicial_laplacian_2 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=2,
                signed=True
            )
            hodge_laplacian_rk2 = simplicial_complex_driver.laplacian(simplicial_laplacian_2)
            logging.info(f'\nHodge-Laplacian rank 2\n{hodge_laplacian_rk2}')
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_adjacency(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=4,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            logging.info(f'\nAdjacency matrix:\n{simplicial_complex_driver.adjacency_matrix()}')
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_incidence_directed_1(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=5,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            for rank in range(0, 3):
                incidence_matrix = simplicial_complex_driver.incidence_matrix(rank=rank)
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{incidence_matrix}')
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_incidence_directed_2(self):
        edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
        face_set = [(2, 3, 4), (1, 2, 3)]
        try:
            simplicial_complex_driver = SimplicialComplexDriver.random(node_feature_dimension=5,
                                                                       edge_node_indices=edge_set,
                                                                       face_node_indices=face_set)
            for rank in range(0, 3):
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{simplicial_complex_driver.incidence_matrix(rank=rank)}')
        except (AssertionError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)



