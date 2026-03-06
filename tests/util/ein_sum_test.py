import unittest
import numpy as np
import logging
import os
import python
from python import SKIP_REASON

class EinSumTest(unittest.TestCase):

    def test_dot(self):
        import torch
        a = np.array([1.0, 0.5, 2.0])
        b = np.array([0.1, 2.0, 1.2])

        # Implicit Einstein summation for dot product
        # c = SUM (a[i].b[i]
        einsum_dot_product = np.einsum('i,i', a, b)
        # Explicit Einstein summation for dot product
        einsum_explicit_dot_product = np.einsum('i,i->', a, b)
        # Using Numpy method
        np_dot = np.dot(a, b)
        logging.info(f'\n{einsum_dot_product=}\n{np_dot=}')
        self.assertTrue(einsum_dot_product == np_dot)
        self.assertTrue(einsum_explicit_dot_product == np_dot)

        # Einsum notation for PyTorch
        einsum_notation = torch.einsum('i,i->',
                                       torch.Tensor(a),
                                       torch.Tensor(b))
        torch_dot = torch.dot(torch.Tensor(a), torch.Tensor(b))
        logging.info(f'{einsum_notation=}\n{torch_dot=}')

    def test_dot_mismatch(self):
        try:
            a = np.array([1.0, 0.5, 2.0])
            b = np.array([0.1, 2.0, 1.2, 1.6])
            einsum_dot_product = np.einsum('i,i', a, b)
            logging.info(einsum_dot_product)
            self.assertTrue(False)
        except Exception as e:
            logging.info(f'Failed with {e}')
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_matrix_mul_numpy(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        b = np.array([[0.1, 2.0], [1.2, 0.5]])

        # Explicit declared the intermediate index k
        # c[i,j] = SUM (a[i,k].b[k,j])
        einsum_mul = np.einsum('ij,jk->ik', a, b)

        # Conventional Matrix and Numpy notation for multiplication
        matrix_mul = a @ b
        matrix_matmul = np.matmul(a, b)
        logging.info(f'\neinsum matrix multiplication:\n{einsum_mul}\nMatrix multiplication:'
                     f'\n{matrix_mul}\nNumpy matmul function:\n{matrix_matmul}')
        self.assertTrue(einsum_mul[0][0] == matrix_mul[0][0])
        self.assertTrue(einsum_mul[1][0] == matrix_matmul[1][0])

    def test_matrix_mul_numpy_2(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        b = np.array([[0.1, 2.0], [1.2, 0.5]])

        # c = a @ b
        c = np.einsum('ij,jk->ik', a, b)
        # Transpose
        d = np.einsum('ij->ji', c)
        logging.info(f'\na @ b:\n{c}\nc.T:\n{d}')
        self.assertTrue(len(c) == len(d))

    def test_matrix_mul_torch(self):
        import torch
        a = torch.Tensor([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])
        b = torch.Tensor([[0.5, -0.5, 0.0],
                      [0.8,  0.1, -0.9],
                      [0.6,  -0.1, 0.5]])

        # Explicit declared the intermediate index k
        # c[i,j] = SUM (a[i,k].b[k,j])
        einsum_matrix_mul = torch.einsum('ij,jk->ik', a, b)

        # Conventional torch function for multiplication
        torch_matrix_mul = torch.mm(a, b)
        logging.info(f'\neinsum matrix multiplication:\n{einsum_matrix_mul}\nTorch mm:\n{torch_matrix_mul}')
        self.assertTrue(einsum_matrix_mul[0][0] == torch_matrix_mul[0][0])

    def test_outer_product(self):
        a = np.array([1.0, 0.5, 2.0, 0.7, 0.9])
        b = np.array([0.1, 2.0, 1.2])

        # Explicit descriptor for outer product of two vectors
        # C[ij] = SUM a[i].b[j]
        einsum_outer_product = np.einsum('i,j -> ij', a, b)
        numpy_outer_product = np.outer(a, b)
        logging.info(f'\neinsum outer product:\n{einsum_outer_product}\nNumpy outer product:\n{numpy_outer_product}')

        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])
        b = np.array([[0.1, 2.0],
                      [1.2, 0.5]])

        # Explicit descriptor for outer product of two matrices
        # C[ijkl] = SUM a[ij].b[kl]
        einsum_outer_product = np.einsum('ij,kl -> ijkl', a, b)

        # Use the Numpy outer function after reshaping
        outer_matrix_shape = einsum_outer_product.shape
        numpy_outer_product = np.outer(a, b).reshape(outer_matrix_shape)
        logging.info(f'\neinsum outer product:\n{einsum_outer_product}\nNumpy outer product:\n{numpy_outer_product}')

    def test_transpose(self):
        from numpy import matrix
        import torch

        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])
        # Explicit descriptor for the transpose of a matrix
        #  A[ji] = transpose(A[ij] )
        einsum_transposed = np.einsum('ij->ji', a)
        # Transpose using Numpy .T operator
        numpy_transpose_operator = a.T

        # Transpose using matrix transpose
        at = a.copy()
        matrix.transpose(at)
        logging.info(f'\nNumpy transpose Einsum:\n{einsum_transposed}\n'
              f'Numpy Transpose operator:\n{numpy_transpose_operator}'
              f'Numpy transpose function:\n{at}')

        self.assertTrue(einsum_transposed[0][1] == numpy_transpose_operator[0][1])
        self.assertTrue(einsum_transposed[1][1] == at[1][1])

        a_tensor = torch.tensor(a)
        einsum_transpose = torch.einsum('ij->ji', a_tensor)
        torch_transpose_operator = a_tensor.T
        at = a.copy()
        torch.transpose(torch.Tensor(at), 0, 1)
        logging.info(f'\neinsum transpose (torch):\n{einsum_transpose}\n'
              f'Transpose operator (torch):\n{torch_transpose_operator}\n'
              f'Transpose function (torch):\n{at}')

    def test_transpose_2(self):
        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])
        #  Transpose
        a_transpose = np.einsum('ij->ji', a)
        # Transpose of transpose
        a_a_transpose = np.einsum('ij->ji', a_transpose)
        logging.info(f'\nTranspose of transpose of:\n{a}\nis:\n{a_a_transpose}')
        self.assertTrue(a[2][2] == a_a_transpose[2][2])
        self.assertTrue(a[1][2] == a_a_transpose[1][2])

    def test_trace(self):
        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])

        # Using the Einstein notation for trace
        # tr(a) = SUM a[ii]
        einsum_trace = np.einsum('ii ->', a)

        # Using the trace method
        numpy_trace = np.trace(a)
        # Manual validation
        manual_trace = sum([a[index, index] for index in range(len(a))])
        logging.info(f'\nTrace Einstein: {einsum_trace}\nTrace numpy: {numpy_trace}'
              f'\nManual trace: {manual_trace}')
        self.assertTrue(einsum_trace == numpy_trace)
        self.assertTrue(einsum_trace == manual_trace)

    @unittest.skip('Ignore')
    def test_kalman_state_eq(self):
        import torch
        # Components of the Kalman State-Transition equation
        A = np.array([[1.0, 0.1], [0.0, 1.0]])  # State transition matrix
        B = np.array([[0.5], [1.0]])  # Control input matrix
        x_k_1 = np.array([2.0, 1.0])  # State vector at time k-1
        u = np.array([0.1])  # Control input
        w = np.array([0.05, 0.02])

        # Using numpy matrix operator
        x_k_from_matrix = A @ x_k_1 + B @ u + w

        # Using einsum implementation in PyTorch
        A_k_x = torch.einsum('ij,j->i',
                             torch.Tensor(A),
                             torch.Tensor(x_k_1))
        B_k_u = torch.einsum('ij,j->i',
                             torch.Tensor(B),
                             torch.Tensor(u))
        x_k_from_einsum = A_k_x + B_k_u + torch.Tensor(w)
        logging.info(f'\nx[k] from matrix computation:\n{x_k_from_matrix}'
              f'\nx[k] from Torch einsum:\n{x_k_from_einsum}')

    @unittest.skip('Ignore')
    def test_neural_linear(self):
        import torch
        W = np.array([[0.20, 0.80, -0.50],
                      [0.50, -0.91, 0.26],
                      [-0.26, 0.27, 0.17]])  # Shape (3, 3)
        x = np.array([1, 2, 3])  # Input vector, shape (3,)
        b = np.array([2.0, 3.0, 0.5])  # Bias vector, shape (3,)

        # Using Numpy @ matrix operator
        y = W @ x + b
        # Using Einstein summation notation in Numpy
        np_einsum_y = np.einsum('ij,j->i', W, x) + b

        # Using Einstein summation notation in PyTorch
        torch_einsum_y = torch.einsum('ij,j->i',
                                      torch.Tensor(W),
                                      torch.Tensor(x)) + torch.Tensor(b)

        logging.info(f'\nNumpy einsum linear transform:\n{np_einsum_y}'
              f'\nTorch einsum linear transform:\n{torch_einsum_y}'
              f'\nDirect linear transform:\n{y}')

    @unittest.skip('Ignore')
    def test_einsum_gradient(self):
        import torch

        # Define a simple function f(x) = x1^2 + x2^3
        def my_f(x: torch.Tensor) -> torch.Tensor:
            return x[0] ** 2 + x[1] ** 3 + 1

        # Arbitrary input tensor
        x = torch.tensor([2.0, 3.0], requires_grad=True)  # Enable gradient tracking

        f = my_f(x)
        # Compute gradient using backward
        f.backward()

        # Compute gradient explicitly using einsum
        gradient_vec = torch.einsum('i->i', x.grad)  # Equivalent to identity mapping
        logging.info("Gradient using einsum:", gradient_vec)  # Output: tensor([4., 27.])

        # Verification
        grad = x.grad
        logging.info("Gradient using autograd:", grad)  # Output: tensor([4., 27.])



