import unittest
from geometry.information_geometry import NormalHypersphere


class NormalHypersphereTest(unittest.TestCase):
    def test_show_points(self):
        normal_dist = NormalHypersphere()
        num_points = 2
        tangent_vector = [0.4, 0.7, 0.2]
        num_manifold_pts = normal_dist.show_points(num_points, tangent_vector)
        self.assertEqual(num_manifold_pts, num_points)

    def test_show_distributions(self):
        normal_dist = NormalHypersphere()
        num_points = 2
        tangent_vector = [0.4, 0.7, 0.2]
        num_manifold_pts = normal_dist.show_points(num_points, tangent_vector)
        self.assertEqual(num_manifold_pts, num_points)
        num_pdfs = 40
        succeeded = normal_dist.show_distribution(num_pdfs, num_points)
        self.assertEqual(succeeded, second=True)

