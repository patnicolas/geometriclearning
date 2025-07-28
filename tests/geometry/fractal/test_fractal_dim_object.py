import unittest
from geometry.fractal import FractalDimObject
import logging
import os
from python import SKIP_REASON

class FractalDimObjectTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        import matplotlib.pyplot as plt

        sample_size = 256
        threshold = 0.85
        fractal_dim_object = FractalDimObject.build(sample_size, threshold, True)
        xyz = fractal_dim_object.xyz
        logging.info(f'len(xyz[1]) = {len(xyz[1])}')

        fig = plt.figure()

        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o')

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add title and labels
        plt.title('Input data pattern')
        plt.show()

        self.assertTrue(fractal_dim_object.xyz.shape[0] == sample_size)
        self.assertTrue(fractal_dim_object.xyz.shape[1] == sample_size)
        self.assertTrue(fractal_dim_object.xyz.shape[2] == sample_size)
        logging.info(str(fractal_dim_object))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_call(self):
        import math
        sample_size = 256
        threshold = 0.92
        fractal_dim_object = FractalDimObject.build(sample_size, threshold, True)
        coefficient, counts, sizes = fractal_dim_object()
        estimated_num_counts = math.log2(sample_size)
        self.assertTrue(len(counts) == estimated_num_counts - 1)
        logging.info(coefficient)
        self.assertTrue( 2 < math.fabs(coefficient) < 3)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_plot(self):
        import matplotlib.pyplot as plt
        import math

        sample_size = 1024
        threshold = 0.88
        fractal_dim_object = FractalDimObject.build(sample_size, threshold)
        _, counts, sizes = fractal_dim_object()
        estimated_num_counts = math.log2(sample_size)
        self.assertTrue(len(counts) == estimated_num_counts-1)

        # Create a scatter plot
        plt.scatter(counts, sizes)

        # Add title and labels
        plt.title('Count vs. sizes counting boxes')
        plt.xlabel('Box counts')
        plt.ylabel('Box sizes')
        plt.legend()
        plt.show()

    def test_4_dim(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import random

        num_pts = 2000
        # Sample data
        x = np.random.rand(num_pts)
        y = np.random.rand(num_pts)
        z = np.random.rand(num_pts)
        levels = np.array([random.gauss(0.5, 0.2) for _ in range(num_pts)])  # Fourth variable for color

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        scatter = ax.scatter(x, y, z, c=levels, cmap='viridis')

        # Add color bar which maps values to colors
        cbar = fig.colorbar(scatter, ax=ax, shrink=1.0, aspect=20)
        cbar.set_label('Color Scale')

        # Labels for axes
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('4 Dimension representation')

        # Show plot
        plt.show()

