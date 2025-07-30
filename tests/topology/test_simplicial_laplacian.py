import unittest
from topology import TopologyException
from topology.simplicial_laplacian import SimplicialLaplacian, SimplicialLaplacianType
import logging
import python


class SimplicialLaplacianTest(unittest.TestCase):

    def test_simplicial_up_laplacian(self):
        try:
            simplicial_laplacian = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.UpLaplacian,
                                                       rank=1,
                                                       signed=True)
            logging.info(simplicial_laplacian)
            edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            face_set = [[1, 2, 3], [2, 3, 4]]
            up_laplacian = simplicial_laplacian(edge_set + face_set)
            logging.info(up_laplacian)
            self.assertTrue(True)
        except TopologyException as e:
            logging.error(e)
            self.assertTrue(False)

    def test_simplicial_down_laplacian(self):
        try:
            simplicial_laplacian = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.DownLaplacian,
                                                       rank=0,
                                                       signed=True)
            edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            face_set = [[1, 2, 3], [2, 3, 4]]
            down_laplacian = simplicial_laplacian(edge_set + face_set)
            logging.info(down_laplacian)
            self.assertTrue(False)
        except TopologyException as e:
            self.assertTrue(True)

    def test_simplicial_hodge_laplacians(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
        face_set = [[1, 2, 3], [2, 3, 4]]

        try:
            for n in range(1, 3):
                simplicial_laplacian = SimplicialLaplacian(simplicial_laplacian_type=SimplicialLaplacianType.HodgeLaplacian,
                                                           rank=1,
                                                           signed=True)
                logging.info(f'Simplicial Hodge-Laplacian rank {n}: {simplicial_laplacian}')
                hodge_laplacian = simplicial_laplacian(edge_set + face_set)
                logging.info(hodge_laplacian)
            self.assertTrue(True)
        except TopologyException as e:
            self.assertTrue(False)
