import unittest
import numpy as np
from topology.simplicial_feature_set import SimplicialFeatureSet
from topology.simplicial_laplacian import SimplicialLaplacian, SimplicialLaplacianType
import logging
import python


class AbstractSimplicialComplexTest(unittest.TestCase):


    @unittest.skip('Ignore')
    def test_init_1(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        face_set = [[2, 3, 4], [1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(4, edge_set, face_set)
            logging.info(simplicial_feature_set)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_2(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        face_set = [[2, 3, 4], [1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(edge_set, face_set)
            logging.info(simplicial_feature_set)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_simplicial_up_laplacian_compute(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
        face_set = [[1, 2, 3], [2, 3, 4]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(5, edge_set, face_set)
            # simplicial_feature_set.show()
            simplicial_laplacian_0 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                         rank=0,
                                                         signed=True)
            up_laplacian_rk0 = simplicial_feature_set.laplacian(simplicial_laplacian_0)
            logging.info(f'\nUP-Laplacian rank 0\n{up_laplacian_rk0}')
            simplicial_laplacian_1 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                         rank=1,
                                                         signed=True)
            up_laplacian_rk1 = simplicial_feature_set.laplacian(simplicial_laplacian_1)
            logging.info(f'\nUP-Laplacian rank 1\n{up_laplacian_rk1}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_simplicial_down_laplacian_compute(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
        face_set = [[1, 2, 3], [2, 3, 4]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(edge_set, face_set)
            simplicial_laplacian_1 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                                                         rank=1,
                                                         signed=True)
            down_laplacian_rk1 = simplicial_feature_set.laplacian(simplicial_laplacian_1)
            logging.info(f'\nDown-Laplacian rank 1\n{down_laplacian_rk1}')
            simplicial_laplacian_2 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                                                         rank=2,
                                                         signed=True)
            down_laplacian_rk2 = simplicial_feature_set.laplacian(simplicial_laplacian_2)
            logging.info(f'\nDown-Laplacian rank 2\n{down_laplacian_rk2}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_simplicial_hodge_laplacian_compute(self):
        edge_set = [[3, 4], [4, 5], [1, 2], [1, 3], [2, 3], [2, 4]]
        face_set = [[2, 3, 4], [1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(5, edge_set, face_set)
            simplicial_laplacian_0 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                                                         rank=0,
                                                         signed=True)
            hodge_laplacian_rk0 = simplicial_feature_set.laplacian(simplicial_laplacian_0)
            logging.info(f'\nHodge-Laplacian rank 0\n{hodge_laplacian_rk0}')

            simplicial_laplacian_1 = SimplicialLaplacian(
                simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                rank=1,
                signed=True)
            hodge_laplacian_rk1 = simplicial_feature_set.laplacian(simplicial_laplacian_1)
            logging.info(f'\nHodge-Laplacian rank 1\n{hodge_laplacian_rk1}')

            simplicial_laplacian_2 = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                                                         rank=2,
                                                         signed=True)
            hodge_laplacian_rk2 = simplicial_feature_set.laplacian(simplicial_laplacian_2)
            logging.info(f'\nHodge-Laplacian rank 2\n{hodge_laplacian_rk2}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_adjacency(self):
        edge_set = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 5]]
        face_set = [[1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(dimension=4, edge_set=edge_set, face_set=face_set)
            logging.info(f'\nAdjacency matrix:\n{simplicial_feature_set.adjacency_matrix()}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_directed_1(self):
        edge_set = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 5]]
        face_set = [[1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(dimension=5, edge_set=edge_set, face_set=face_set)
            for rank in range(0, 3):
                incidence_matrix = simplicial_feature_set.incidence_matrix(rank=rank)
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{incidence_matrix}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_directed_2(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        face_set = [[2, 3, 4], [1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet.build(dimension=5, edge_set=edge_set, face_set=face_set)
            for rank in range(0, 4):
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{simplicial_feature_set.incidence_matrix(rank=rank)}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_undirected(self):
        features = np.array([
            [0.4, 0.6, -0.8, 0.1],      # node 1
            [0.0, -0.3, 0.2, -0.5],     # node 2
            [0.5, -0.3, 0.0, -0.1],     # node 3
            [0.8, 0.1, -0.7,  0.0],     # node 4
            [0.3, -0.4, -0.2, 1.0],     # node 5
        ])
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        face_set = [[2, 3, 4], [1, 2, 3]]
        try:
            simplicial_feature_set = SimplicialFeatureSet(features, edge_set, face_set)
            for rank in range(0, 3):
                undirected_incidence = simplicial_feature_set.incidence_matrix(rank=rank, directed_graph=False)
                logging.info(f'\nUndirected incidence matrix rank {rank}:\n{undirected_incidence}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)



