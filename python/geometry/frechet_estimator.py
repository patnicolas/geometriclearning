__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geomstats.geometry.manifold import Manifold
from geomstats.learning.frechet_mean import FrechetMean, BaseGradientDescent
from torch import Tensor
import numpy as np
from typing import List, AnyStr
from geometry import GeometricException


class FrechetEstimator(object):
    def __init__(self, space: Manifold, optimizer: BaseGradientDescent, weights: Tensor = None) -> None:
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

    def estimate(self, X: List[np.array]) -> np.array:
        if len(X) < 1:
            raise GeometricException('Frechet estimator: At least one tensor does not belong to the manifold')

        def estimate_step(X: List[np.array]) -> np.array:
            if len(X) > 1:
                sub_mean = []
                for idx in range(len(X)-1):
                    stacked = np.stack(arrays=[X[idx], X[idx+1]], axis=0)
                    self.frechet_mean.fit(X=stacked, y=None, weights=self.weights)
                    z = self.frechet_mean.estimate_
                    sub_mean.append(z)
                return estimate_step(sub_mean)
            else:
                return X[0]
        return Tensor(estimate_step(X=X))

    def rand(self, num_samples: int) -> List[np.array]:
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
        return [x for x in X if self.__belongs(x)]

    @staticmethod
    def euclidean_mean(manifold_points: List[Tensor]) -> np.array:
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
        assert x.shape[0] == 3, f'Point {x} should have 3 dimension'
        """
        Test if a point belongs to this hypersphere
        @param point defined as a list of 3 values
        @return True if the point belongs to the manifold, False otherwise
        """
        return self.space.belongs(x)









