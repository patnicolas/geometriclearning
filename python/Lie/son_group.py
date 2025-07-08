__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import torch
import numpy as np
from typing import AnyStr, Dict, List
from lie import LieException
__all__ = ['SOnGroup']


class SOnGroup(object):

    def __init__(self, dim: int, equip: bool, atol: float = 1e-5) -> None:
        """
        Constructor for the SO(n) lie Group of Rotation matrices of dimension 2, 3 or 4
        @param dim: Dimension of lie algebra
        @type dim: int
        @param equip: Flag to specify if this lie Group is equipped with a Riemannian metric
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
        Compute the lie algebra (Rotation, skewed matrices) from a point on a manifold. The method leverages Geomstats
        SpecialOrthogonal.lie_algebra method.
        
        @param point: Point on the manifold
        @type point: torch Tensor
        @param identity: Reference point on the lie Algebra (Tangent space). Default identity matrix is used if
                        the argument is not provided
        @type identity: torch Tensor | None
        @return: Rotation matrix associated with the point on the manifold
        @rtype: torch Tensor
        """
        self.__validate_tensor_shape(point, dim=self.__group.n)

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
        points = torch.Tensor(self.__group.n, n_samples)
        SOnGroup.validate_points(points, dim=self.__group.n, rtol=self.__atol)
        return points

    def random_matrix(self) -> torch.Tensor:
        """
        Generate a random rotation matrix of group SO(n) for n = 2,3 & 4
        @return: Random rotation matrix
        @rtype: Torch tensor
        """
        A = torch.rand(self.__group.n, self.__group.n)
        Q, R = torch.linalg.qr(A)
        # Force det(Q) = +1 to preserve orientation
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    def generate_rotation(self, weights: List[float]) -> torch.Tensor:
        """
        Generate a rotation from a weighted sum of basis matrices with sum(weights) = 1.0
        The number of basis matrices for so2 is 1, so3 3 and so4 6
        math:
            A_{so3} = \sum_{i=0}^{3} w_{i}\begin{vmatrix} \
                        0 & -\alpha_{i} &  \beta_{i} \\ \
                        \alpha_{i} & 0 & -\gamma_{i} \\ \
                        -\beta_{i} & \gamma_{i} & 0 \
                        \end{vmatrix}  \    \     \     \alpha_{i} = (-1, 1), \beta_{i}=(-1, 1), \gamma_{i}=(-1, 1)
        A Geometric Exception is thrown if number of weights does not match number of basis matrices

        @param weights: Weight distribution
        @type weights: List[float]
        @return: Torch tensor of shape (dim, dim)
        @rtype: torch.Tensor
        """
        basis_matrices = [v for _, v in SOnGroup.__basis_matrices(self.__group.n).items()]
        if len(weights) != len(basis_matrices):
            raise LieException(
                f'Number of weights {len(weights)} should match '
                f'number of basis matrices {len(basis_matrices)}'
            )

        # If there is more than one basis matrices...
        if len(weights) > 1:
            return sum([weights[idx]*torch.Tensor(mat) for idx, mat in enumerate(basis_matrices)])
        else:
            _, basic_matrix = next(iter(basis_matrices))
            return weights[0]*torch.Tensor(basic_matrix)

    def num_basis_matrices(self) -> int:
        """
        Return the number of basis matrices on lie algebra for SO(n)

        @return:  Number of basis matrices
        @rtype: int
        """
        basic_matrices_length = {2: 1, 3: 3, 4: 6}
        return basic_matrices_length[self.__group.n]

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
        SOnGroup.validate_points(points, dim=self.__group.n, rtol=self.__atol)

        # Test only the first element if dimension or 3
        pt = points[0] if len(points.shape) > 2 else points
        return self.__group.belongs(pt.numpy(), self.__atol)

    @staticmethod
    def validate_points(*points: torch.Tensor, dim: int, rtol: float = 1e-5) -> None:
        """
        Validate the matrix on the SO(n) group in 3 steps
        1- Correct shape
        2- Orthogonality   R^T.R = Id
        3- Orientation preservation   det(R) = +1
        A GeometricException is thrown if one of the 3 conditions is not met

        @param points: One or more matrices or points on SO(n) group to be validated
        @type points: torch.Tensor
        @param dim: Dimension of the group (size of rotation matrix)
        @type dim: int
        @param rtol: Error tolerance for validating the condition on preserving the orientation
        @type rtol: float
        """
        from python import are_tensors_close

        for pt in points:
            _shape = (pt.shape[1], pt.shape[2]) if len(pt.shape) > 2 else pt.shape
            # Validate of tensor shape
            if _shape != (dim,  dim):
                raise LieException(f'Shape tensor {_shape} should match({dim}, {dim})')

            # Validate condition of orthogonality   R^T.R = Identity
            if not are_tensors_close(pt.T @ pt, torch.eye(dim), rtol=rtol):
                raise LieException(f'Orthogonality R^T.R  failed')

            # Validate orientation det(R) = +1
            det = torch.linalg.det(pt)
            if abs(det - 1.0) > rtol:
                raise LieException(f'Determinant {det} should be +1')

    def exp(self, tgt_vector: torch.Tensor, base_point: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the end point of a tangent vector given a base point on the manifold.
            end_point = base_point + exp(tgt_vector).
        The method leverages Geomstats SpecialOrthogonal.exp function.
        
        @param tgt_vector: Vector (Matrix lie Algebra) on the tangent space
        @type tgt_vector: torch Tensor
        @param base_point: Base point on the manifold
        @type base_point: torch Tensor
        @return: End point on the manifold
        @rtype: torch Tensor
        """
        # First  Validate shape, Orthogonality and Determinant condition
        base = base_point.numpy() if base_point is not None else np.eye(self.__group.n)
        SOnGroup.validate_points(base, dim=self.__group.n, rtol=self.__atol)
        # Invoke Geomstats library
        end_point = torch.Tensor(self.__group.exp(tgt_vector.numpy(), base))
        # Validate the output as SO(n) rotation
        SOnGroup.validate_points(end_point, dim=self.__group.n, rtol=self.__atol)
        return end_point

    def log(self, point: torch.Tensor, base_point: torch.Tensor= None) -> torch.Tensor:
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
        base = base_point.numpy() if base_point is not None else np.eye(self.__group.n)
        # Validate the input
        SOnGroup.validate_points(point, base, dim=self.__group.n, rtol=self.__atol)
        # Invoke Geomstats
        tgt_vector = torch.Tensor(self.__group.log(point.numpy(), base.numpy()))
        return tgt_vector

    def compose(self, point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
        """
        Compose (or multiply) two points or rotations on SO(n) group. A Geometric exception is raised if one
        of the two input rotation matrices does not belong to so3 or if the composed matrix does not belong to so3

        @param point1: First rotation
        @type point1: torch Tensor
        @param point2: Second rotation
        @type point2: torch Tensor
        @return: Composed rotations which is also a SO(n) group element
        @rtype: torch Tensor
        """
        # We make sure that the two rotations are actually SO(n) elements
        SOnGroup.validate_points(point1, point2, dim=self.__group.n, rtol=self.__atol)
        composed_points = self.__group.compose(point1, point2)
        # Validate the output as SO(n) element
        SOnGroup.validate_points(composed_points, dim=self.__group.n, rtol=self.__atol)
        return composed_points

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
        SOnGroup.validate_points(point, dim=self.__group.n, rtol=self.__atol)
        match self.__group.n:
            # Projection computed using Geomstats
            case 2 | 3:
                inverse_point = torch.Tensor(self.__group.inverse(point.numpy()))
                SOnGroup.validate_points(inverse_point, dim=self.__group.n, rtol=self.__atol)
                return inverse_point
            # Homegrown computation
            case 4:
                return point.T
            case _:
                raise GeometricException(f'Dimension {self.__group.n} is not supported')

    def project(self, point: torch.Tensor) -> torch.Tensor:
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
        from python import are_tensors_close
        return are_tensors_close(t1, t2, rtol=rtol)

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
                'A1': np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
                'A2': np.array([[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]]),
                'A3': np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                'B1': np.array([[0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
                'B2': np.array([[0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 1, 0, 0]]),
                'B3': np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
            }
            case _:
                raise LieException(f'SO({dim}) is not supported')

    def __validate(self, *points: torch.Tensor) -> None:
        SOnGroup.validate_points(*points, dim=self.__group.n, rtol=self.__atol)

    @staticmethod
    def __validate_tensor_shape(*points: torch.Tensor, dim: int) -> None:
        for x in points:
            sh = (x.shape[1], x.shape[2]) if len(x.shape) > 2 else x.shape
            if sh != (dim,  dim):
                raise LieException(f'Shape tensor {sh} should match({dim}, {dim})')

    @staticmethod
    def __projection_so4(point: torch.Tensor) -> torch.Tensor:
        import numpy as np

        U, _, Vtrans = np.linalg.svd(point.numpy())
        V = [1, 1, 1, np.linalg.det(U @ Vtrans)]
        R = U @ np.diag(V) @ Vtrans
        return torch.Tensor(R)
