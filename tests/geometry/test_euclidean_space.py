import unittest
import logging
import python
from geometry.euclidean_space import EuclideanSpace
from geometry.geometric_space import GeometricSpace
from geometry.visualization.space_visualization import VisualizationParams


class TestEuclideanSpace(unittest.TestCase):

    def test_sample_2_euclidean(self):
        dim = 2
        num_samples = 5
        euclidean_space = EuclideanSpace(dim)
        logging.info(str(euclidean_space))
        euclidean_data = euclidean_space.sample(num_samples)
        logging.info(f'\n{euclidean_data=}')

    def test_sample_3_euclidean(self):
        dim = 3
        num_samples = 100
        euclidean_space = EuclideanSpace(dim)
        euclidean_data = euclidean_space.sample(num_samples)
        logging.info(f'\n{euclidean_data=}')

    def test_euclidean_mean(self):
        dim = 2
        num_samples = 20
        euclidean_space = EuclideanSpace(dim)
        data = euclidean_space.sample(num_samples)
        average_points = GeometricSpace.euclidean_mean(data)
        logging.info(f'\n{average_points=}')

    def test_euclidean_3d_visualization(self):
        style = {'color': 'blue', 'linestyle': '--', 'label': '2'}
        dim = 3
        num_samples = 8
        euclidean_space = EuclideanSpace(dim)
        data = euclidean_space.sample(num_samples)
        visualParams = VisualizationParams("Hypersphere 3D display", "locations", (8, 8), style, "3d")
        EuclideanSpace.show(visualParams, data)


if __name__ == '__main__':
    unittest.main()