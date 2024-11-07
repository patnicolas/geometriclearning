import unittest
from informationgeometry.beta_hypersphere import BetaHypersphere


class BetaHypersphereTest(unittest.TestCase):
    def test_show_distribution(self):
        beta_dist = BetaHypersphere()
        num_interpolations = 10
        num_manifold_pts = 200
        beta_dist.show_points(num_interpolations)
        succeeded = beta_dist.show_distribution(num_manifold_pts, num_interpolations)
        self.assertEqual(succeeded, second=True)