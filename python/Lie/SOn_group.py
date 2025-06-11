__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import torch
import numpy as np
from typing import AnyStr, Dict
from geometry import GeometricException



class SOnGroup(object):

    def __init__(self, dim: int, point_type: AnyStr, equip: bool, atol: float = 1e-5) -> None:
        """
        Constructor for the SO(n) Lie Group of Rotation matrices of dimension 2, 3 or 4
        @param dim: Dimension of Lie algebra
        @type dim: int
        @param point_type: Type of point (Vector or rotation matrix)
        @type point_type: str
        @param equip: Flag to specify if this Lie Group is equipped with a Riemannian metric
        @type equip: bool
        @param atol: Error tolerance for tensor value
        @type atol: float
        """
        assert dim in (2, 3, 4), f'Dimension of SO({dim}) group is not supported'
        self.__atol = atol
        # Invoke Geomstats Special Orthogonal group
        self.__group = SpecialOrthogonal(n=dim, point_type=point_type, equip=equip)

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
        identity = identity.numpy() if identity is not None else np.eye(self.__group.n)
        return torch.Tensor(self.__group.lie_algebra(point.numpy(), identity))

    def sample_points(self, n_samples: int) -> torch.Tensor:
        """
        Generate one or several random points on the manifold (uniform distribution). The method leverages Geomstats 
        SpecialOrthogonal.random_uniform function.
        
        @param n_samples: Number of samples
        @type n_samples: int
        @return: Tensor containing the points on the SO(n) manifold
        @rtype: torch Tensor
        """
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
        return self.__group.belongs(points.numpy(), self.__atol).all()

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
        return torch.Tensor(self.__group.log(point.numpy(), base_point.numpy()))

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
