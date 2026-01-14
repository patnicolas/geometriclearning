import unittest
import logging
import python
from geometry.discrete import WassersteinException
from geometry.discrete.olliver_ricci import OlliverRicci
import torch


class OlliverRicciTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_create_adjacency(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
            adjacency = OlliverRicci.create_adjacency(edge_index)
            logging.info(f'Adjacency:\n{adjacency}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_curvature_1(self):
        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

            r = torch.tensor([0.1, 0.1, 0.2, 0.4], dtype=torch.float32)
            c = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float32)
            olliver_ricci = OlliverRicci(edge_index=edge_index, weights=None, epsilon=0.05, rc=(r, c))
            curvature = olliver_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
            logging.info(f'\nCurvature with r, c:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_curvature_3(self):
        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            weights = torch.Tensor([1.0, 4.0, 0.5, 1.6, 2.6, 1.5])
            r = torch.tensor([0.1, 0.1, 0.2, 0.4], dtype=torch.float32)
            c = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float32)

            olliver_ricci = OlliverRicci(edge_index=edge_index, weights=weights, epsilon=0.05, rc=(r, c))
            curvature = olliver_ricci.curvature(n_iters=100, early_stop_threshold=0.0001)
            logging.info(f'\nCurvature with r, c and weights:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    def test_curvature_5(self):
        def sphere_geodesic(n_edges: int) -> torch.Tensor:
            import math

            weights = []
            delta = math.pi / (8 * n_edges)
            start_geodesic = 0.0
            for i in range(n_edges):
                lat1, lon1 = map(math.radians, [0.0, 0.0])
                lat2, lon2 = map(math.radians, [start_geodesic, start_geodesic + delta])

                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                weights.append(2.0 * math.asin(math.sqrt(a)))
                start_geodesic += delta
            return torch.Tensor(weights)

        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            r = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.1], dtype=torch.float32)
            c = torch.tensor([0.1, 0.1, 0.4, 0.2, 0.2], dtype=torch.float32)

            olliver_ricci = OlliverRicci.build(edge_index=edge_index,
                                               geodesic_distance=sphere_geodesic,
                                               epsilon=0.05,
                                               rc=(r, c))
            curvature = olliver_ricci.curvature(n_iters=100, early_stop_threshold=0.0001)
            logging.info(f'\nCurvature:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_curvature_2(self):
        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            olliver_ricci = OlliverRicci(edge_index=edge_index, weights=None, epsilon=0.05, rc=None)

            curvature = olliver_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
            logging.info(f'\nCurvature from joint:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_curvature_4(self):
        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            olliver_ricci = OlliverRicci(edge_index=edge_index, weights=None, epsilon=0.05, rc=None)

            curvature = olliver_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
            logging.info(f'\nCurvature from joint large:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)
