import unittest
import logging
import python
from geometry.discrete import WassersteinException
from geometry.discrete.ollivier_ricci import OllivierRicci
import torch


class OllivierRicciTest(unittest.TestCase):
    @unittest.skip('Ignored')
    def test_create_adjacency(self):
        try:
            edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
            adjacency = OllivierRicci.create_adjacency(edge_index)
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
            ollivier_ricci = OllivierRicci(edge_index=edge_index, weights=None, epsilon=0.05, rc=(r, c))
            curvature = ollivier_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
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

            ollivier_ricci = OllivierRicci(edge_index=edge_index, weights=weights, epsilon=0.05, rc=(r, c))
            curvature = ollivier_ricci.curvature(n_iters=100, early_stop_threshold=0.0001)
            logging.info(f'\nCurvature with r, c and weights:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    # @unittest.skip('Ignored')
    def test_curvature_5(self):
        def sphere_geodesics(n_edges: int) -> torch.Tensor:
            import math

            weights = []
            delta = math.pi / (6 * n_edges)
            geo_x = 3*math.pi/16
            geo_y = math.pi/6
            for i in range(n_edges):
                lat1, lon1 = map(math.radians, [geo_x, geo_y])
                lat2, lon2 = map(math.radians, [geo_x - delta, geo_y + delta])

                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(0.5*dlat) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(0.5*dlon) ** 2
                weights.append(2.0 * math.asin(math.sqrt(a)))
                if i % 2 == 0:
                    geo_y -= delta
                else:
                    geo_x -= delta
                    geo_y += delta
            return torch.Tensor(weights)

        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            r = torch.tensor([0.1, 0.1, 0.2, 0.4, 0.0], dtype=torch.float32)
            c = torch.tensor([0.1, 0.5, 0.0, 0.2, 0.2], dtype=torch.float32)

            ollivier_ricci = OllivierRicci.build(edge_index=edge_index,
                                                 geodesic_distance=sphere_geodesics,
                                                 epsilon=0.05,
                                                 rc=(r, c))
            curvature = ollivier_ricci.curvature(n_iters=100, early_stop_threshold=0.0001)
            logging.info(f'\nCurvature:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_curvature_2(self):
        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            ollivier_ricci = OllivierRicci(edge_index=edge_index, weights=None, epsilon=0.05, rc=None)

            curvature = ollivier_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
            logging.info(f'\nCurvature from joint:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_curvature_4(self):
        try:
            edge_index = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            edge_weights = torch.Tensor([0.8, 1.5, 2.6, 4.8, 2.2, 2.5, 6.1, 0.1, 3.8, 3.5])
            ollivier_ricci = OllivierRicci(edge_index=edge_index, weights=edge_weights, epsilon=0.02, rc=None)

            curvature = ollivier_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
            logging.info(f'\nCurvature from joint large:\n{curvature}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)
