import unittest
import numpy as np
from util.ein_sum import EinSum


class EinSumTest(unittest.TestCase):

    def test_dot(self):
        a = np.array([1.0, 0.5, 2.0])
        b = np.array([0.1, 2.0, 1.2])
        ein_sum = EinSum(a, b)
        dot_value = ein_sum.dot()
        ref_dot_value = np.dot(a, b)
        print(f'\nDot product Einstein sum {dot_value}\nDirect product {ref_dot_value}')
        self.assertTrue(dot_value == ref_dot_value)

    def test_matrix_mul(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        b = np.array([[0.1, 2.0], [1.2, 0.5]])
        ein_sum = EinSum(a, b)
        output_matrix = ein_sum.matrix_mul()
        ref_matrix = a @ b
        print(f'\nMatrix multiplication Einstein:\n{output_matrix}\nDirect multiplication:\n{ref_matrix}')
        self.assertTrue(output_matrix[0][0] == ref_matrix[0][0])
        self.assertTrue(output_matrix[1][0] == ref_matrix[1][0])

    def test_matrix_el_mul(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        b = np.array([[0.1, 2.0], [1.2, 0.5]])
        ein_sum = EinSum(a, b)
        output = ein_sum.matrix_el_sum()
        output_matrix = a * b
        ref_output = np.sum(output_matrix)
        print(f'\nMatrix el. multiplication Einstein:\n{output}\Multiplication numpy:\n{ref_output}')
        self.assertTrue(output == ref_output)

    def test_outer_product(self):
        a = np.array([1.0, 0.5, 2.0, 0.7])
        b = np.array([0.1, 2.0, 1.2])
        ein_sum = EinSum(a, b)
        output = ein_sum.outer_product()
        ref_output = np.outer(a, b)
        print(f'\nOuter product Einstein:\n{output}\nOuter product numpy:\n{ref_output}')
        self.assertTrue(output[0][1] == ref_output[0][1])

    def test_transpose(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        ein_sum = EinSum(a)
        output = ein_sum.transpose()
        ref_output = a.T
        print(f'\nTranspose Einstein:\n{output}\nTranspose numpy:\n{ref_output}')
        self.assertTrue(output[0][1] == ref_output[0][1])
        self.assertTrue(output[1][1] == ref_output[1][1])

    def test_trace(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        ein_sum = EinSum(a)
        output = ein_sum.trace()
        ref_output = np.trace(a)
        print(f'\nTrace Einstein:\n{output}\nTrace numpy:\n{ref_output}')
        self.assertTrue(output == ref_output)

    def test_kalman_state_eq(self):
        A = np.array([[1.0, 0.1], [0.0, 1.0]])  # State transition matrix
        B = np.array([[0.5], [1.0]])  # Control input matrix
        x_k_1 = np.array([2.0, 1.0])  # State vector at time k-1
        u = np.array([0.1])  # Control input
        w = np.array([0.05, 0.02])

        # Using numpy matrix operator
        ref_x_k = A @ x_k_1 + B @ u + w
        # Using Einstein summation method
        x_k = np.einsum('ij,j->i', A, x_k_1) + np.einsum('ij,j->i', B, u) + w
        print(f'\nEinstein state eq:\n{x_k}\nDirect state eq:\n{ref_x_k}')
        self.assertTrue(x_k[0] == ref_x_k[0])

    def test_neural_linear(self):
        W = np.array([[0.20, 0.80, -0.50],
                      [0.50, -0.91, 0.26],
                      [-0.26, 0.27, 0.17]])  # Shape (3, 3)
        x = np.array([1, 2, 3])  # Input vector, shape (3,)
        b = np.array([2.0, 3.0, 0.5])  # Bias vector, shape (3,)

        # Using @ operator
        ref_y = W @ x + b
        # Using Einstein summation
        y = np.einsum('ij,j->i', W, x) + b
        print(f'\nEinstein linear transform:\n{y}\nDirect linear transform:\n{ref_y}')




