__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr
import numpy as np
import geomstats.backend as gs
from geomstats.geometry.functions import HilbertSphere
from geometry.manifold_point import ManifoldPoint
from geometry.geometric_exception import GeometricException


"""
    Class wrapper for the Function space using the Hilbert Sphere. The constructor generates the
    sample for the domain associated with the functions. The Hilbert domain is defined as [0, 1].
    This class inherit the Hilbert Sphere defined in the Geomstats module
    
    :param num_domain_samples Number of samples (equidistant) used to represent the Hilbert interval
    Throw a GeometricException if the number of samples < 2
"""


class FunctionSpace(HilbertSphere):
    def __init__(self, num_domain_samples: int):
        if num_domain_samples < 2:
            raise GeometricException(f'Number of samples {num_domain_samples} should be > 1')

        domain_samples = gs.linspace(0, 1, num=num_domain_samples)
        super(FunctionSpace, self).__init__(domain_samples, True)

    def create_manifold_point(self, id: AnyStr, vector: np.array, base_point: np.array) -> ManifoldPoint:
        """
            Generate a manifold pont with a base_point of the manifold and a direction, vector
            :param id Identifier for the Manifold Point
            :param vector A vector that define the direction of the tangent vector
            :param base_point Point or anchor on the manifold
            :throw GeometricException If the base point does not belong to the Hilbert space
        """
        if not self.belongs(base_point):
            raise GeometricException(f'{base_point} does not belong to this function sphere space')

        # Compute the tangent vector using the direction 'vector' and point 'base_point'
        tgt_vector = self.to_tangent(vector, base_point)
        return ManifoldPoint(id, base_point, tgt_vector)

    def random_manifold_points(self, n_samples: int) -> List[ManifoldPoint]:
        """
            Generate a list of n_samples of random points on the Hilbert sphere
            :param n_samples Number of random points on the Hilbert sphere
            :throw GeometricException If the Number of random points is not positive
        """
        if n_samples < 1:
            raise GeometricException(f'Number of random points {n_samples} should be >0')

        return [ManifoldPoint(
            id=f'rand_{n+1}',
            location=random_pt) for n, random_pt in enumerate(self.random_point(n_samples))]

    def exp(self, vector: np.array, manifold_base_pt: ManifoldPoint) -> np.array:
        """
            Exponential map.
            Compute the Riemannian exponential of the vector 'vector' onto the Hilbert Sphere.
            :param vector  Defines the direction of the projection
            :param manifold_base_pt The point on the Hilbert sphere to be projected
            :return Exponential or projection of the vector onto the Hilbert Sphere
        """
        if not self.belongs(manifold_base_pt.location):
            raise GeometricException(f'{manifold_base_pt.id} does not belong to this function sphere space')

        return self.metric.exp(tangent_vec=vector, base_point=manifold_base_pt.location)

    def log(self, manifold_base_pt: ManifoldPoint, target_pt: ManifoldPoint) ->np.array:
        """
            Inverse of the exponential map.
            Compute the Riemannian logarithm of a given base point from a target point
            :param manifold_base_pt The point on the Hilbert sphere
            :param target_pt The point on the Hilbert sphere used to compute the loga
            :return  The tangent vector at 'manifold_base_pt' equal to the Riemannian logarithm
                    of 'target_pt' at the base point.
        """
        if not self.belongs(manifold_base_pt.location):
            raise GeometricException(f'{manifold_base_pt.id} does not belong to this function sphere space')
        if not self.belongs(target_pt.location):
            raise GeometricException(f'{target_pt.id} does not belong to this function sphere space')

        return self.metric.log(point=manifold_base_pt.location, base_point=target_pt.location)

    def inner_product(self, tgt_vector1: np.array, tgt_vector2: np.array) -> np.array:
        """
            Compute the inner product of two tangent vectors
            :param tgt_vector1 First tangent vector
            :param tgt_vector2 Second tangent vector
            :return Inner product of the two tangent vectors
            :throw GeometricException if the tangent vectors have different length
            """
        if len(tgt_vector1) is not len(tgt_vector2):
            raise GeometricException(f'Length tgt vector1 {len(tgt_vector1)} != length tgt vector2 {len(tgt_vector2)}')
        return self.metric.inner_product(tgt_vector1,tgt_vector2)

    def manifold_point_inner_product(self, manifold_base_pt: ManifoldPoint, manifold_pt: ManifoldPoint) -> np.array:
        """
            Compute the inner product of two tangent vector associated with two manifold points
            :param manifold_base_pt Base point on Hilbert sphere with a defined tangent vector
            :param manifold_pt Second point on Hilbert sphere with a defined tangent vector
            :return Inner product of the two tangent vectors associated with the manifold points
            :throw GeometricException if tangent vectors are undefined or have different length
        """
        if manifold_base_pt.tgt_vector is None:
            raise GeometricException(f'Tangent vector for {manifold_base_pt.id} is undefined')
        if manifold_pt.tgt_vector is None:
            raise GeometricException(f'Tangent vector for {manifold_pt.id} is undefined')
        if len(manifold_base_pt.tgt_vector) is not len(manifold_pt.tgt_vector):
            raise GeometricException(f'Length tgt vector1 {len(manifold_base_pt.tgt_vector)} != length tgt vector2 '
                                     f'{len(manifold_pt.tgt_vector)}')

        return self.metric.inner_product(
            manifold_base_pt.tgt_vector,
            manifold_pt.tgt_vector,
            manifold_base_pt.location)

    def __str__(self):
        return f'Domain: {self.domain}\nShape: {self.shape}\nDimension: {self.dim}\nMetric: {self.metric}'
