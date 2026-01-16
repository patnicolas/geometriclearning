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
from typing import List, Optional
# 3rd Party imports
from geomstats.geometry.manifold import Manifold
from geomstats.learning.frechet_mean import FrechetMean, BaseGradientDescent
from torch import Tensor
import numpy as np

from geometry import GeometricException
__all__ = ['FrechetEstimator']


class FrechetEstimator(object):
    def __init__(self, space: Manifold, optimizer: BaseGradientDescent, weights: Optional[Tensor] = None) -> None:
        """
        Constructor for the Frechet estimator. The instance relies on the Geomstats FrechetMean class
        @param space: Manifold for which the Frechet mean is computed
        @type space: Manifold
        @param optimizer: Optimizer used in the computation of the mean
        @type optimizer: Sub-class of BaseGradientDescent
        @param weights: Weighting of various data points in the computation of the mean
        @type weights: Torch tensor
        """
        self.frechet_mean = FrechetMean(space)
        self.frechet_mean.optimizer = optimizer
        self.weights = weights
        self.space = space

    def estimate(self, X: np.array) -> np.array:
        self.frechet_mean.fit(X=X, y=None, weights=self.weights)
        return self.frechet_mean.estimate_

    def rand(self, num_samples: int) -> np.array:
        """
        Generate num_samples random value on a given manifold specified in the constructor (space)
        @param num_samples: Number of samples
        @type num_samples: int
        @return: List of Numpy arrays representing the various random point on the manifold
        @rtype: List
        """
        if num_samples < 2:
            raise GeometricException(f'Need at least 2 data point to compute the Frechet mean')
        from geomstats.geometry.hypersphere import Hypersphere
        from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices

        # Test if this manifold is supported
        if not (isinstance(self.space, Hypersphere) or isinstance(self.space, _SpecialOrthogonalMatrices)):
            raise GeometricException('Cannot generate random values on unsupported manifold')
        X = self.space.random_uniform(num_samples)

        # Validate the randomly generated belongs to the manifold 'self.space'
        are_points_valid = all([self.space.belongs(x) for x in X])
        if not are_points_valid:
            raise GeometricException('Some generated points do not belong to the manifold')
        return X

    @staticmethod
    def euclidean_mean(manifold_points: np.array) -> np.array:
        """
        Compute the Euclidean mean for a set of data point on a manifold
        @param manifold_points List of manifold data points
        @type manifold_points: List Numpy arrays
        @return the mean value as a np vector
        @rtype Numpy array
        """
        return np.mean(manifold_points, axis=0)

    """ ----------------------------   Private helper methods --------------------------- """

    def __belongs(self, x: np.array) -> bool:
        """
        Test if a point belongs to this hypersphere
        @param x defined as a list of 3 values
        @return True if the point belongs to the manifold, False otherwise
        """
        if x.shape[0] != 3:
            raise ValueError(f'Point {x} should have 3 dimension')
        return self.space.belongs(x)









