import unittest
import torch
from geometry.discrete.floyd_warshall import FloydWarshall
import logging
import python

class FloydWarshallTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_init_1(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
            weights = torch.Tensor([
                0.1, 0.4, 0.2, 0.1, 0.5, 0.7, 1.0, 0.6, 0.3
            ])

            floyd_warshall = FloydWarshall(edge_index, weights)
            logging.info(floyd_warshall)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_init_failed(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
            weights = torch.Tensor([
                0.1, 0.4, 0.2, 0.1, 0.5, 0.7, 1.0, 0.6
            ])

            floyd_warshall = FloydWarshall(edge_index, weights)
            logging.info(floyd_warshall)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skip('Ignored')
    def test_init_2(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
            floyd_warshall = FloydWarshall(edge_index)
            logging.info(floyd_warshall)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skip('Ignored')
    def test_call(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
            weights = torch.Tensor([
                0.1, 0.4, 0.2, 0.1, 0.5, 0.7, 1.0, 0.6, 0.3
            ])
            floyd_warshall = FloydWarshall(edge_index, weights)
            logging.info(f'\n{floyd_warshall()}')
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skip('Ignored')
    def test_call_2(self):
        try:
            edge_index = [(0, 1), (0, 3), (1, 2), (2, 3)]
            weights = torch.Tensor([
                0.0, 5.0, 3.0, 1.0
            ])

            floyd_warshall = FloydWarshall(edge_index, weights)
            logging.info(f'\n{floyd_warshall()}')
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_call_3(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]

            def sphere_geodesic(n_edges: int) -> torch.Tensor:
                import math

                weights = []
                delta = math.pi/(8*n_edges)
                start_geodesic = 0.0
                for i in range(n_edges):
                    lat1, lon1 = map(math.radians, [0.0, 0.0])
                    lat2, lon2 = map(math.radians, [start_geodesic, start_geodesic+delta])

                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                    weights.append(2.0 * math.asin(math.sqrt(a)))
                    start_geodesic += delta
                return torch.Tensor(weights)

            logging.info(sphere_geodesic(8))
        except ValueError as e:
            logging.error(e)
            self.assertTrue(True)


