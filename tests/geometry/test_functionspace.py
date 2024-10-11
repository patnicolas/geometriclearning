import unittest
from manifolds.functionspace import FunctionSpace
import numpy as np


class TestFunctionSpace(unittest.TestCase):
    @unittest.skip('ignore')
    def test_init(self):
        num_samples = 100
        function_space = FunctionSpace(num_samples)
        print(str(function_space))

    # @unittest.skip('ignore')
    def test_create_manifold_point(self):
        num_samples = 4
        function_space = FunctionSpace(num_samples)
        random_pt = function_space.random_point()
        print(random_pt)
        vector = np.array([1.0, 0.5, 1.0, 0.0])
        assert(num_samples == len(vector))
        manifold_pt = function_space.create_manifold_point('id', vector, random_pt)
        print(f'Manifold point: {str(manifold_pt)}')

    def test_random_manifold_points(self):
        num_hilbert_samples = 8
        function_space = FunctionSpace(num_hilbert_samples)
        n_points = 5
        random_manifold_pts = function_space.random_manifold_points(n_points)
        print(random_manifold_pts[0])

    def test_exp(self):
        num_Hilbert_samples = 8
        function_space = FunctionSpace(num_Hilbert_samples)

        vector = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        assert num_Hilbert_samples == len(vector)
        exp_map_pt = function_space.exp(vector, function_space.random_manifold_points(1)[0])
        print(f'Exponential on Hilbert Sphere: {str(exp_map_pt)}')

    def test_log(self):
        num_Hilbert_samples = 8
        function_space = FunctionSpace(num_Hilbert_samples)

        random_points = function_space.random_manifold_points(2)
        log_map_pt = function_space.log(random_points[0], random_points[1])
        print(f'Logarithm from Hilbert Sphere {str(log_map_pt)}')

    def test_inner_product_same_vector(self):
        import math

        num_Hilbert_samples = 8
        function_space = FunctionSpace(num_Hilbert_samples)
        vector = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        inner_prd = function_space.inner_product(vector, vector)
        print(f'Euclidean norm of vector: {np.linalg.norm(vector)}')
        print(f'Inner product of same vector: {str(inner_prd)}')
        print(f'Norm of vector: {str(math.sqrt(inner_prd))}')

    def test_inner_product_different_vectors(self):
        num_Hilbert_samples = 8
        functions_space = FunctionSpace(num_Hilbert_samples)
        vector1 = np.array([0.5, 1.0, 0.0, 0.4, 0.7, 0.6, 0.2, 0.9])
        vector2 = np.array([0.5, 0.5, 0.2, 0.4, 0.6, 0.6, 0.5, 0.5])
        inner_prd = functions_space.inner_product(vector1, vector2)
        print(f'Inner product of different vectors: {str(inner_prd)}')


if __name__ == '__main__':
    unittest.main()
