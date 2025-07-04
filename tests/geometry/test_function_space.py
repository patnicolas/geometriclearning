import unittest
from geometry.function_space import FunctionSpace
import numpy as np
import logging
import os
import python
from python import SKIP_REASON


class TestFunctionSpace(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        num_samples = 100
        function_space = FunctionSpace(num_samples)
        logging.info(str(function_space))

    def test_create_manifold_point(self):
        num_samples = 4
        function_space = FunctionSpace(num_samples)
        random_pt = function_space.random_point()
        logging.info(random_pt)
        vector = np.array([1.0, 0.5, 1.0, 0.0])
        assert(num_samples == len(vector))
        manifold_pt = function_space.create_manifold_point('id', vector, random_pt)
        logging.info(f'Manifold point: {str(manifold_pt)}')

    def test_random_manifold_points(self):
        num_hilbert_samples = 8
        function_space = FunctionSpace(num_hilbert_samples)
        n_points = 5
        random_manifold_pts = function_space.random_manifold_points(n_points)
        logging.info(random_manifold_pts[0])

    def test_exp(self):
        num_Hilbert_samples = 8
        function_space = FunctionSpace(num_Hilbert_samples)

        vector = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        assert num_Hilbert_samples == len(vector)
        exp_map_pt = function_space.exp(vector, function_space.random_manifold_points(1)[0])
        logging.info(f'Exponential on Hilbert Sphere: {str(exp_map_pt)}')

    def test_log(self):
        num_Hilbert_samples = 8
        function_space = FunctionSpace(num_Hilbert_samples)

        random_points = function_space.random_manifold_points(2)
        log_map_pt = function_space.log(random_points[0], random_points[1])
        logging.info(f'Logarithm from Hilbert Sphere {str(log_map_pt)}')

    def test_inner_product_same_vector(self):
        import math

        num_Hilbert_samples = 8
        function_space = FunctionSpace(num_Hilbert_samples)
        vector = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        inner_prd = function_space.inner_product(vector, vector)
        logging.info(f'Euclidean norm of vector: {np.linalg.norm(vector)}')
        logging.info(f'Inner product of same vector: {str(inner_prd)}')
        logging.info(f'Norm of vector: {str(math.sqrt(inner_prd))}')

    def test_inner_product_different_vectors(self):
        num_Hilbert_samples = 8
        functions_space = FunctionSpace(num_Hilbert_samples)
        vector1 = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        vector2 = np.array([0.5, 0.5, 0.2, 0.4, 0.6, 0.6, 0.5, 0.5])
        inner_prd = functions_space.inner_product(vector1, vector2)
        logging.info(f'Inner product of different vectors: {str(inner_prd)}')


if __name__ == '__main__':
    unittest.main()
