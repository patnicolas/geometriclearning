import unittest
from topology.complex_laplacian import ComplexLaplacian
from topology import LaplacianType
import logging
import python


class ComplexLaplacianTest(unittest.TestCase):

    def test_simplicial_up_laplacian(self):
        try:
            simplicial_laplacian = ComplexLaplacian(laplacian_type=LaplacianType.UpLaplacian,
                                                    rank=1,
                                                    signed=True)
            logging.info(simplicial_laplacian)
            edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            face_set = [[1, 2, 3], [2, 3, 4]]
            up_laplacian = simplicial_laplacian(edge_set + face_set)
            logging.info(f'Up Laplacian:\n{up_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_simplicial_down_laplacian(self):
        try:
            simplicial_laplacian = ComplexLaplacian(laplacian_type=LaplacianType.DownLaplacian,
                                                    rank=0,
                                                    signed=True)
            edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            face_set = [[1, 2, 3], [2, 3, 4]]
            down_laplacian = simplicial_laplacian(edge_set + face_set)
            logging.info(f'{down_laplacian}')
            self.assertTrue(False)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(True)

    def test_simplicial_down_laplacian_2(self):
        try:
            simplicial_laplacian = ComplexLaplacian(laplacian_type=LaplacianType.DownLaplacian,
                                                    rank=2,
                                                    signed=True)
            edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            face_set = [[1, 2, 3], [2, 3, 4]]
            down_laplacian = simplicial_laplacian(edge_set + face_set)
            logging.info(f'Down Laplacian:\n{down_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_simplicial_hodge_laplacian(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
        face_set = [[1, 2, 3], [2, 3, 4]]

        try:
            for n in range(1, 3):
                simplicial_laplacian = ComplexLaplacian(laplacian_type=LaplacianType.HodgeLaplacian,
                                                        rank=1,
                                                        signed=True)
                logging.info(f'Simplicial Hodge-Laplacian rank {n}: {simplicial_laplacian}')
                hodge_laplacian = simplicial_laplacian(edge_set + face_set)
                logging.info(f'Hodge Laplacian:\n{hodge_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)
