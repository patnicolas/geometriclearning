import unittest
from geometry.kendall_space import KendallSpace
from geometry.visualization.space_visualization import VisualizationParams
import logging

class TestKendallSpace(unittest.TestCase):

    def test_sample_kendall_sphere(self):
        num_samples = 20
        manifold = KendallSpace()
        data = manifold.sample(num_samples)
        logging.info(f'Kendall:\n{str(data)}')

    def test_kendall_sphere(self):
        num_samples = 2
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = KendallSpace()
        logging.info(str(manifold))
        data = manifold.sample(num_samples)
        visualParams = VisualizationParams("Data on Kendall", "Data on Kendall S32", (8, 8))
        KendallSpace.show(visualParams, data, 'S32')


if __name__ == '__main__':
    unittest.main()