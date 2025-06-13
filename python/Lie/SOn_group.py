__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import torch
import numpy as np
from typing import AnyStr, Dict
from geometry import GeometricException



class SOnGroup(object):

    def __init__(self, dim: int, equip: bool, atol: float = 1e-5) -> None:
        """
        Constructor for the SO(n) Lie Group of Rotation matrices of dimension 2, 3 or 4
        @param dim: Dimension of Lie algebra
        @type dim: int
        @param equip: Flag to specify if this Lie Group is equipped with a Riemannian metric
        @type equip: bool
        @param atol: Error tolerance for tensor value
        @type atol: float
        """
        assert dim in (2, 3, 4), f'Dimension of SO({dim}) group is not supported'
        assert 1e-8 <= atol <= 1e-3, f'Atol argument {atol} should be [1e-8, 1e-3]'

        self.__atol = atol
        # Invoke Geomstats Special Orthogonal group
        self.__group = SpecialOrthogonal(n=dim, point_type='matrix', equip=equip)

    def __str__(self) -> AnyStr:
        so_type = f'SO({self.__group.n})'
        is_equipped = 'with' if hasattr(type(self.__group), 'metric')  else 'without'
        representation = 'vector' if len(self.__group.identity) == 1 else 'matrix'
        basis_matrices = '\n'.join([f'{k}: {str(v)}' for k, v in SOnGroup.__basis_matrices(self.__group.n).items()])
        return f'{so_type}, representation: {representation}, {is_equipped} metric\n{basis_matrices}'

    def lie_algebra(self, point: torch.Tensor, identity: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the Lie algebra (Rotation, skewed matrices) from a point on a manifold. The method leverages Geomstats 
        SpecialOrthogonal.lie_algebra method.
        
        @param point: Point on the manifold
        @type point: torch Tensor
        @param identity: Reference point on the Lie Algebra (Tangent space). Default identity matrix is used if
                        the argument is not provided
        @type identity: torch Tensor | None
        @return: Rotation matrix associated with the point on the manifold
        @rtype: torch Tensor
        """
        self.__validate_tensor_shape(point)

        # Use the default identity if none is provided
        identity = identity.numpy() if identity is not None else np.eye(self.__group.n)
        # The logarithm map is need to compute the tangent vector from identity to the current point
        rot_matrix = self.__group.log(point.numpy(), identity)
        return torch.Tensor(rot_matrix)

    def sample_points(self, n_samples: int) -> torch.Tensor:
        """
        Generate one or several random points on the manifold (uniform distribution). The method leverages Geomstats 
        SpecialOrthogonal.random_uniform function.
        
        @param n_samples: Number of samples
        @type n_samples: int
        @return: Tensor containing the points on the SO(n) manifold
        @rtype: torch Tensor
        """
        assert 0 < n_samples < 1024, f'Number of random samples {n_samples} should be [1, 1024['
        return torch.Tensor(self.__group.random_uniform(n_samples))

    def belongs(self, points: torch.Tensor) -> bool:
        """
        Test if a given set of points belongs to the manifold. The method leverages Geomstats SpecialOrthogonal.belongs
        function.
        
        @param points: Points on the manifold
        @type points: Torch Tensor
        @return: True if all the points belongs to the manifold, False otherwise
        @rtype: bool
        """
        # Validate shape, Orthogonality and Determinant condition
        self.__validate(points)

        # Test only the first element if dimension or 3
        pt = points[0] if len(points.shape) > 2 else points
        return self.__group.belongs(pt.numpy(), self.__atol)

    @staticmethod
    def validate_son_input(*points: torch.Tensor, dim: int, rtol: float = 1e-5) -> None:
        SOnGroup.__validate_tensor_shape(*points, dim=dim)
        for pt in points:
            det = torch.linalg.det(pt)
            if abs(det - 1.0) > 1e-5:
                raise GeometricException(f'Determinant {det} should be 1')
            # id = pt.T @ pt
            if not torch.allclose(pt.T @ pt, torch.eye(dim), rtol=rtol):
                raise GeometricException(f'Orthogonality R^T.R  failed')

    def exp(self, tgt_vector: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        """
        Compute the end point of a tangent vector given a base point on the manifold.
            end_point = base_point + exp(tgt_vector).
        The method leverages Geomstats SpecialOrthogonal.exp function.
        
        @param tgt_vector: Vector (Matrix Lie Algebra) on the tangent space
        @type tgt_vector: torch Tensor
        @param base_point: Base point on the manifold
        @type base_point: torch Tensor
        @return: End point on the manifold
        @rtype: torch Tensor
        """
        # First  Validate shape, Orthogonality and Determinant condition
        self.__validate(tgt_vector, base_point)
        return torch.Tensor(self.__group.exp(tgt_vector.numpy(), base_point.numpy()))

    def log(self, point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        """
        Compute the tangent vector given a base point and an end point on the manifold.
            tangent_vector = log(end_point) - base_point.
        The method leverages Geomstats SpecialOrthogonal.log function.
        
        @param point: End point on the manifold
        @type point: torch Tensor
        @param base_point: Base point on the manifold
        @type base_point: torch Tensor
        @return: Vector as a matrix, tensor
        @rtype: torch Tensor
        """
        # First  Validate shape, Orthogonality and Determinant condition
        self.__validate(point, base_point)
        return torch.Tensor(self.__group.log(point.numpy(), base_point.numpy()))

    def inverse(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a point on a SO(n) rotation group
        math::
            inv(M) = M^{T}

        @param point: SO(n) element for which the inverse has to be computed
        @type point: torch Tensor
        @return: Inverse rotation matrix
        @rtype: torch Tensor
        """
        # First  Validate shape, Orthogonality and Determinant condition
        self.__validate(point)
        match self.__group.n:
            # Projection computed using Geomstats
            case 2 | 3:
                return torch.Tensor(self.__group.inverse(point.numpy()))
            # Homegrown computation
            case 4:
                return point.T
            case _:
                raise GeometricException(f'Dimension {self.__group.n} is not supported')

    def projection(self, point: torch.Tensor) -> torch.Tensor:
        """
        The projection of n x n matrix onto SO(n) refers to finding the closest matrix in SO(n). The dimension
        n = 2 and 3 are supported by Geomstats SpecialOrthogonal.projection function and n =4 relies on homegrown
        implementation.
        math::
            M\in \mathbb{R}^{n \times  n}  \  \  \  \exists R\in SO(n)  \  \
            R = \arg \min \left\{  \left\| M - R \right\|_{F} =\sum_{i=1}^{n} \sum_{j=1}^{m}|M_{ii}-R_{ij}|\right\}

        @param point: Matrix to be projected
        @type point: Torch Tensor
        @return: Projected matrix
        @rtype: Torch Tensor
        """
        #  Validate shape, Orthogonality and Determinant condition
        self.__validate(point)
        match self.__group.n:
            # Projection computed using Geomstats
            case 2 | 3: return torch.Tensor(self.__group.projection(point.numpy()))
            # Homegrown computation
            case 4: return SOnGroup.__projection_so4(point)
            case _:
                raise GeometricException(f'Dimension {self.__group.n} is not supported')

    def equal(self, t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-6) -> bool:
        """
        Test if two tensors are similar for which the values are close within a margin of error:
            abs({t1}i - {t2}i) < atol + rtol*abs({t2}i)
            
        @param t1: First tensor
        @type t1: torch tensor
        @param t2: Second tensor
        @type t2: torch tensor
        @param rtol: Error tolerance for elements of the second tensor
        @type rtol: float
        @return: True if the two tensors are similar, False otherwise
        @rtype: bool
        """
        #  Validate shape, Orthogonality and Determinant condition
        self.__validate(t1, t2)
        return torch.allclose(t1, t2, rtol=rtol, atol=self.__atol)

    """ ------------------------   Private Helper Methods -------------------------------- """
    
    @staticmethod
    def __basis_matrices(dim: int) -> Dict[AnyStr, np.array]:
        match dim:
            case 2: return {
                'E': np.array([[0, -1], [1, 0]])
            }
            case 3: return {
                'F1': np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
                'F2': np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                'F3': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
            }
            case 4: return {
                'A1': np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
                'A2': np.array([[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]]),
                'A3': np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                'B1': np.array([[0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
                'B2': np.array([[0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 1, 0, 0]]),
                'B3': np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]])
            }
            case _:
                raise GeometricException(f'SO({dim}) is not supported')

    def __validate(self, *points: torch.Tensor) -> None:
        SOnGroup.validate_son_input(*points, dim=self.__group.n, rtol=self.__atol)

    @staticmethod
    def __validate_tensor_shape(*points: torch.Tensor, dim: int) -> None:
        for x in points:
            sh = (x.shape[1], x.shape[2]) if len(x.shape) > 2 else x.shape
            if sh != (dim,  dim):
                raise GeometricException(f'Shape tensor {sh} should match({dim}, {dim})')

    @staticmethod
    def __projection_so4(matrix: torch.Tensor) -> torch.Tensor:
        import numpy as np

        U, _, Vtrans = np.linalg.svd(matrix.numpy())
        V = [1, 1, 1, np.linalg.det(U @ Vtrans)]
        R = U @ np.diag(V) @ Vtrans
        return torch.Tensor(R)
