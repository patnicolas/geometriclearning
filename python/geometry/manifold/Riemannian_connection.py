__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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


# Standard Library imports
from typing import AnyStr, Optional, List
# 3rd Party imports
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.base import LevelSet
import numpy as np
# Library imports
from geometry.manifold.manifold_point import ManifoldPoint
from geometry import GeometricException
__all__ = ['RiemannianConnection']

class RiemannianConnection(object):
    """
        Create a formal Riemann connection with associated metric from a Manifold or Level set.
        The metric is automatically selected from the type of manifold. This class is a wrapper of
        the Connection, Manifold and RiemannianMetric classes in Geomstats library

        Initialization sequence:   Manifold (space) => Metric => Connection
     """
    def __init__(self, space: LevelSet, manifold_type: AnyStr):
        """
        Constructor for this Riemannian manifold

        @param space: Manifold used for the connection
        @type space: LevelSet inheriting from Geomstats Manifold class
        @param manifold_type: type of manifold used as descriptor (default None)
        @type manifold_type: str
        """
        self.riemannian_metric = RiemannianConnection.__get_metric(space)
        self.manifold_descriptor = f'{manifold_type}\nDimension: {space.dim}\nShape: {space.shape}' \
                                   f'\nCoordinates type: {space.default_coords_type}'
        # Add 1 dimension for extrinsic coordinates
        self.ndim = space.dim + 1 if space.default_coords_type == "extrinsic" else space.dim

    def __str__(self):
        return f'Riemannian Connection for {self.manifold_descriptor}'

    def inner_product(self, tgt_vec1: np.array, tgt_vec2: np.array, base_pt: np.array) -> np.array:
        """
        Compute the inner product of two tangent vector at a given base point on a manifold.
        A GeometricException is thrown if the vector have different sizes

        :param tgt_vec1: First tangent vector
        :type tgt_vec1: Numpy array
        :param tgt_vec2: Second tangent vector
        :type tgt_vec2:  Numpy array
        :param base_pt: Base point on the Manifold
        :type base_pt:  Numpy array
        :return: inner product in a range [-1, 1] if both vector are not null, 0 otherwise
        :rtype:  Numpy array
        """
        if len(tgt_vec1) != len(tgt_vec2):
            raise GeometricException(f'Inner product of vector size {len(tgt_vec1)} and vector size {len(tgt_vec1)}')

        return 0 if len(tgt_vec1) == 0 or len(tgt_vec2) == 0 \
            else self.riemannian_metric.inner_product(tgt_vec1, tgt_vec2, base_pt)

    @staticmethod
    def euclidean_inner_product(vector1: np.array, vector2: np.array) -> np.array:
        """
        Compute the Euclidean inner product of two vectors using Numpy function
        :param vector1: First vector of the Euclidean inner product
        :type vector1: Numpy array
        :param vector2: Second vector of the Euclidean inner product
        :type vector2: Numpy array
        :return: Numpy inner product in range [-1, 1] if none of the vectors are 0, 0 otherwise
        :rtype: Numpy array
        """
        if len(vector1) != len(vector2):
            raise GeometricException(f'Inner product of vector size {len(vector1)} and vector size {len(vector2)}')

        return 0 if len(vector1) == 0 or len(vector2) == 0 else np.inner(vector1, vector2)

    def manifold_point_inner_product(
            self,
            manifold_base_pt: ManifoldPoint,
            manifold_pt: ManifoldPoint) -> np.array:
        """
        Compute the inner product of two instance of ManifoldPoint
        :param manifold_base_pt: Base manifold point structure
        :type manifold_base_pt: ManifoldPoint
        :param manifold_pt: Second manifold point structure
        :type manifold_pt: ManifoldPoint
        :return: inner product in a range [0, 1] if both tangent vectors are not null, 0 otherwise
        :rtype: Numpy/float value
        """
        return self.inner_product(manifold_base_pt.tgt_vector, manifold_pt.tgt_vector, manifold_base_pt.location)

    def norm(self, vector: np.array, base_pt: np.array) -> np.array:
        """
        Compute the norm of a vector on a manifold (extrinsic coordinates) given a base point on the manifold
        :param vector: Vector for which the norm has to be computed
        :type vector: Numpy array
        :param base_pt: Base point on the manifold
        :type base_pt: Numpy array
        :return: Norm of the vector at the given base point
        :rtype: Numpy array
        """
        if len(vector) != len(base_pt):
            raise GeometricException(f'Norm of vector size {len(vector)} at base point of size {len(base_pt)}')

        return 0 if len(vector) == 0 else self.riemannian_metric.norm(vector, base_pt)

    def manifold_point_norm(self, manifold_base_pt: ManifoldPoint) -> np.array:
        return self.norm(manifold_base_pt.tgt_vector, manifold_base_pt.location)

    def parallel_transport(
            self,
            manifold_base_pt: ManifoldPoint,
            manifold_end_pt: Optional[ManifoldPoint] = None,
            direction: Optional[np.array] = None) -> np.array:
        """
        Compute the parallel transport of a tangent vector from a base point in a Manifold point instance.
        :param manifold_base_pt: Manifold base point (defined as ManifoldPoint instance)
        :type manifold_base_pt:  ManifoldPoint
        :param manifold_end_pt: Optional end point for parallel transport from the manifold base point
        :type manifold_end_pt:  ManifoldPoint
        :param direction: Direction for the vector fields associated with the parallel transport
        :type direction: Numpy array
        :return: Parallel transport
        :rtype: Numpy array
        """
        if manifold_base_pt.ndim() != self.ndim:
            raise GeometricException(f'Base pt dimension {manifold_base_pt.ndim()} should be {self.ndim}')

        return self.riemannian_metric.parallel_transport(
            manifold_base_pt.tgt_vector,
            manifold_base_pt.location,
            direction,
            manifold_end_pt.location)

    def levi_civita_coefficients(self, base_pt: np.array) -> np.array:
        """
        Compute the coefficients of the Levi-Civita connection (or Christoffel symbols) on a Riemann manifold
        which metric is extracted from the hypersphere
        :param base_pt:
        :type base_pt: Numpy array
        :return: The intrinsic metric for the Christoffel symbols
        :rtype: Numpy array
        """
        if len(base_pt) != self.ndim:
            raise GeometricException(f'Base pt dimension {len(base_pt)} should be {self.ndim}')
        return self.riemannian_metric.christoffels(base_pt)

    def curvature_tensor(self, tgt_vectors: List[np.array], base_pt: np.array) -> np.array:
        """
        Compute the curvature tensor based on Christoffel symbols for a manifold using Levi-Civita connection
        The computation of the curvature tensor relies on 3 indices tensor or tangent vectors.
        A Geometric exception is raised if the tangent vectors are not properly defined.

        :param tgt_vectors: List of 3 tangent vectors along the curve on the manifold
        :type tgt_vectors: List of Numpy array
        :param base_pt: Base point on the manifold
        :type base_pt: Numpy array
        :return: Curvature tensor
        :rtype: Numpy array
        """
        if len(tgt_vectors) != 3 or any(vec is None for vec in tgt_vectors):
            raise GeometricException(f'Tangent vectors for the curvature {str(tgt_vectors)} is not properly defined')

        return self.riemannian_metric.curvature(tgt_vectors[0], tgt_vectors[1], tgt_vectors[2], base_pt)

    def curvature_derivative_tensor(self, tgt_vectors: List[np.array], base_pt: Optional[np.array] = None) -> np.array:
        """
        Compute the derivative of the curvature tensor on a Riemann manifold equipped with a tensor metric
        and Levi-Civita connection.
        :param tgt_vectors: List of 4 tangent vector with dimension ndim and dimension of base point
        :type tgt_vectors: List of Numpy Array
        :param base_pt: Base point on the Riemann manifold
        :type base_pt: Numpy array
        :return: Curvature derivative tensor
        :rtype:Numpy array
        """
        if len(tgt_vectors) != 4 or any(vec is None or len(vec) != len(tgt_vectors[0]) for vec in tgt_vectors):
            raise GeometricException(f'Tangent vectors for curvature derivative are undefined')
        if len(base_pt) != len(tgt_vectors[0]):
            raise GeometricException(f'Dimension of base point {len(base_pt)} should be equal to dimension tangent '
                                     f'vectors {len(tgt_vectors[0])}')

        return self.riemannian_metric.curvature_derivative(
            tgt_vectors[0],
            tgt_vectors[1],
            tgt_vectors[2],
            tgt_vectors[3],
            base_pt)

    def sectional_curvature_tensor(self,
                                   tgt_vec1: np.array,
                                   tgt_vec2: np.array,
                                   base_pt: Optional[np.array] = None) -> np.array:
        """

        :param tgt_vec1:
        :type tgt_vec1:
        :param tgt_vec2:
        :type tgt_vec2:
        :param base_pt:
        :type base_pt:
        :return:
        :rtype:
        """
        if len(tgt_vec1) != len(tgt_vec2):
            raise GeometricException(f'Dimension of tangent vectors for sectional curvature {len(tgt_vec1)} '
                                     f'and {len(tgt_vec2)} should be identical')

        return self.riemannian_metric.sectional_curvature(tgt_vec1, tgt_vec2, base_pt)

    def geodesic(self, manifold_base_pt: ManifoldPoint, end_point: ManifoldPoint = None) -> np.array:
        return NotImplementedError("method not implemented")

        # ----------------------  Helper methods -------------------------

    @staticmethod
    def __get_metric(space: LevelSet) -> RiemannianMetric:
        from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
        from geomstats.geometry.poincare_ball import PoincareBall, PoincareBallMetric
        from geomstats.geometry.hyperboloid import Hyperboloid

        match space:
            case Hypersphere():
                return HypersphereMetric(space)
            case PoincareBall(space.dim):
                return PoincareBallMetric(space)
            case Hyperboloid(space.dim):
                return Hypberboloid(space)
            case _:
                raise NotImplementedError(f'Manifold {str(space)} not supported')
