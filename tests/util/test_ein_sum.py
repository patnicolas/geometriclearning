import unittest
import numpy as np
from util.ein_sum import EinSum


class EinSumTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_dot(self):
        import torch
        a = np.array([1.0, 0.5, 2.0])
        b = np.array([0.1, 2.0, 1.2])

        # Implicit Einstein summation for dot product
        # c = SUM (a[i].b[i]
        implicit_notation = np.einsum('i,i', a, b)
        # Explicit Einstein summation for dot product
        explicit_notation = np.einsum('i,i->', a, b)
        # Using Numpy method
        np_dot = np.dot(a, b)
        print(f'\nDot product Einstein notation {explicit_notation}\nNumpy dot {np_dot}')
        self.assertTrue(explicit_notation == np_dot)
        self.assertTrue(implicit_notation == np_dot)

        # Einsum notation for PyTorch
        einsum_notation = torch.einsum('i,i->',
                                       torch.Tensor(a),
                                       torch.Tensor(b))
        torch_dot = torch.dot(torch.Tensor(a), torch.Tensor(b))
        print(f'Einsum notation: {einsum_notation}\nTorch dot: {torch_dot}')

    @unittest.skip('Ignore')
    def test_matrix_mul_numpy(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        b = np.array([[0.1, 2.0], [1.2, 0.5]])

        # Explicit declared the intermediate index k
        # c[i,j] = SUM (a[i,k].b[k,j])
        explicit_notation = np.einsum('ij,jk->ik', a, b)

        # Conventional Matrix and Numpy notation for multiplication
        matrix_notation = a @ b
        numpy_notation = np.matmul(a, b)
        print(f'\nExplicit notation:\n{explicit_notation}\nMatrix notation:\n{matrix_notation}')
        self.assertTrue(explicit_notation[0][0] == matrix_notation[0][0])

    @unittest.skip('Ignore')
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
        einsum_notation = torch.einsum('ij,jk->ik', a, b)

        # Conventional torch function for multiplication
        torch_function = torch.mm(a, b)
        print(f'\nExplicit notation:\n{einsum_notation}\nMatrix notation:\n{torch_function}')
        self.assertTrue(einsum_notation[0][0] == torch_function[0][0])

    @unittest.skip('Ignore')
    def test_matrix_el_mul(self):
        a = np.array([[1.0, 0.5], [2.0, 1.5]])
        b = np.array([[0.1, 2.0], [1.2, 0.5]])
        ein_sum = EinSum(a, b)
        output = ein_sum.matrix_el_sum()
        output_matrix = a * b
        ref_output = np.sum(output_matrix)
        print(f'\nMatrix el. multiplication Einstein:\n{output}\Multiplication numpy:\n{ref_output}')
        self.assertTrue(output == ref_output)

    @unittest.skip('Ignore')
    def test_outer_product(self):
        a = np.array([1.0, 0.5, 2.0, 0.7, 0.9])
        b = np.array([0.1, 2.0, 1.2])

        # Explicit descriptor for outer product of two vectors
        # C[ij] = SUM a[i].b[j]
        explicit_notation = np.einsum('i,j -> ij', a, b)
        numpy_function = np.outer(a, b)
        print(f'\nOuter Product Vector Einstein:\n{explicit_notation}\nOuter product numpy:\n{numpy_function}')

        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])
        b = np.array([[0.1, 2.0],
                      [1.2, 0.5]])

        # Explicit descriptor for outer product of two matrices
        # C[ijkl] = SUM a[ij].b[kl]
        explicit_notation = np.einsum('ij,kl -> ijkl', a, b)

        # Use the Numpy outer function after reshaping
        outer_matrix_shape = explicit_notation.shape
        numpy_function = np.outer(a, b).reshape(outer_matrix_shape)
        print(f'\nOuter product Matrix Einstein:\n{explicit_notation}\nOuter product numpy:\n{numpy_function}')


    @unittest.skip('Ignore')
    def test_transpose(self):
        from numpy import matrix
        import torch

        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])

        # Explicit descriptor for the transpose of a matrix
        #  A[ji] = transpose(A[ij] )
        np_einsum_transposed = np.einsum('ij->ji', a)

        # Transpose using Numpy .T operator
        np_transpose_operator = a.T

        # Transpose using matrix transpose
        at = a.copy()
        matrix.transpose(at)
        print(f'\nNumpy transpose Einsum:\n{np_einsum_transposed}\n'
              f'Numpy Transpose operator:\n{np_transpose_operator}'
              f'Numpy transpose function:\n{at}')

        self.assertTrue(np_einsum_transposed[0][1] == np_transpose_operator[0][1])
        self.assertTrue(np_einsum_transposed[1][1] == at[1][1])

        a_tensor = torch.tensor(a)
        torch_einsum_transposed = torch.einsum('ij->ji', a_tensor)
        torch_transpose_operator = a_tensor.T
        at = a.copy()
        torch.transpose(torch.Tensor(at), 0, 1)
        print(f'\nTorch transpose Einsum:\n{torch_einsum_transposed}\n'
              f'Torch Transpose operator:\n{torch_transpose_operator}\n'
              f'Torch transpose function:\n{at}')




    @unittest.skip('Ignore')
    def test_trace(self):
        a = np.array([[1.0, 0.5, 0.4],
                      [0.5, -0.3, 0.2],
                      [-1.0, 0.4, -0.3]])

        # Using the Einstein notation for trace
        # tr(a) = SUM a[ii]
        Einstein_notation = np.einsum('ii ->', a)

        # Using the trace method
        numpy_function = np.trace(a)

        # Manual validation
        tr = sum([a[index, index] for index in range(len(a))])

        print(f'\nTrace Einstein:\n{Einstein_notation}\nTrace numpy:\n{ref_output}')
        self.assertTrue(Einstein_notation == numpy_function)


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


        print(f'\nx[k] from matrix computation:\n{x_k_from_matrix}'
              f'\nx[k] from Torch einsum:\n{x_k_from_einsum}')
       # self.assertTrue(x_k_from_einsum[0] == x_k_from_matrix[0])


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

        print(f'\nNumpy einsum linear transform:\n{np_einsum_y}'
              f'\nTorch einsum linear transform:\n{torch_einsum_y}'
              f'\nDirect linear transform:\n{y}')




