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

from geomstats.information_geometry.base import InformationManifoldMixin
from informationgeometry.cf_statistical_manifold import CFStatisticalManifold
from typing import Tuple
import torch

from geometry import GeometricException
ParamType = torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"


class FisherRao(CFStatisticalManifold):
    """
    Class that wraps the computation of the Riemannian metric for some common statistical manifolds and support the
    computation of:
    - Riemannian metric (Fisher_Rao matrix) [Using Geomstats]
    - Inner product of distribution parameters on tangent space  [Homegrown implementation]
    - Distance between two distributions parameters [Homegrown implementation]

    The Riemannan metric for a distribution on a manifold is defined by
    math::
           g_{j k}(\theta)=\int_X \frac{\partial \log p(x, \theta)}
                {\partial \theta_j}\frac{\partial \log p(x, \theta)}
                {\partial \theta_k} p(x, \theta) d x

    The metric matrix for a distribution with pdf f, and parmeters \theta is defined as
    math::
                  I_{ij} = \int \
                        \partial_{i} f_{\theta}(x)\
                        \partial_{j} f_{\theta}(x)\
                        \frac{1}{f_{\theta}(x)}

    The inner-product of two vectors or tensors in the tangent space at a given parameter point on the statistical
    manifold is defined as
    math::
                  \partial_k I_{ij} = \int \
                        \partial_{ki}^2 f\partial_j f \frac{1}{f} + \
                        \partial_{kj}^2 f\partial_i f \frac{1}{f} - \
                        \partial_i f \partial_j f \partial_k f \frac{1}{f^2}

    """

    def __init__(self, info_manifold: InformationManifoldMixin, bounds: Tuple[float, float]) -> None:
        """
        Constructor for the computation of the Fisher_Rao metric for the most common family of univariate and
        bivariate distributions
        @param info_manifold: Information manifold or Class of distribution
        @type info_manifold: A sub-class of InformationManifoldMixin such as ExponentialDistributions or
        GeometricDistributions
        @param bounds: Tuple of values which set the bounds of input to probability density function
        @type bounds: Tuple[float, float]
        """
        super(FisherRao, self).__init__(info_manifold, bounds)

    def distance(self, point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance between two distributions on the manifold given parameters theta1 (point 1)
        and theta2 (point 2).
        The types of point 1 and point 2 are torch.Tensor for single parameters distributions supported by the
        base class StatisticalManifolds
        @param point1: Parameters for the first distribution
        @type point1: torch.Tensor or Tuple of torch.Tensor
        @param point2:  Parameters for the second distribution
        @type point2: torch.Tensor or Tuple of torch.Tensor
        @return: Distance between two points on the statistical manifold
        @rtype: torch.Tensor
        """
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return torch.abs(torch.log(point2) - torch.log(point1))
            case 'GeometricDistributions':
                return 2.0*torch.abs(torch.arctanh(torch.sqrt(1 - point1) - torch.arctanh(torch.sqrt(1 - point2))))
            case 'PoissonDistributions':
                return 2.0 * torch.abs(torch.sqrt(point2) - torch.sqrt(point1))
            case 'BinomialDistributions':
                import math
                n = self.info_manifold.n_draws
                m = math.sqrt(n) * torch.abs(torch.arcsin(2 * point2 - 1) - torch.arcsin(2 * point1 - 1))
                return m
            case _:
                raise GeometricException(f'Distance for {self.info_manifold.__class__.__name__} not supported')

    def inner_product(self, point: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute the inner product of two vector u, v at a given point on the statistical manifold
        @param point: Point of the closed-form statistical manifold
        @type point: torch.Tensor
        @param v: First vector
        @type v: torch.Tensor
        @param w: Second vector
        @type w: torch.Tensor
        @return: Inner product of two vectors on a given point on manifold
        @rtype: torch.Tensor
        """
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return v*w / (point ** 2)
            case 'PoissonDistributions':
                return v * w/point
            case 'GeometricDistributions':
                g = 1 / point ** 2 + 1 / (1 - point) ** 2
                return g*v*w
            case 'BinomialDistributions':
                n = self.info_manifold.n_draws
                return n*v*w/(point - point*point)
            case _:
                raise GeometricException(f'inner product for {self.info_manifold.__class__.__name__} not supported')

    def norm(self, point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of a vector at a given point on a statistical manifold
        @param point: Point of the closed-form statistical manifold
        @type point: torch.Tensor
        @param v: vector
        @type v: torch.Tensor
        @return: Inner product of two vectors on a given point on manifold
        @rtype: torch.Tensor
        """
        return torch.sqrt(self.inner_product(point, v, v))

    """  -----------------------  Private Helpers methods ------------------   """


    @staticmethod
    def __distance_univariate_normal(point1: Tuple[torch.Tensor, torch.Tensor],
                                     point2: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu1 = point1[0]
        sigma1 = point1[1]
        mu2 = point2[0]
        sigma2 = point2[1]
        if sigma1 < 1e-12 or sigma2 < 1e-12:
            raise GeometricException(f'Normal distribution sigma {torch.min(sigma1, sigma2)} out of bounds')
        t1 = (mu1 - mu2) ** 2  # (mu1 - mu2)**2
        t2 = (sigma1 - sigma2) ** 2  # (sigma1 - sigma2)**2
        c = 1 + 0.5 * (t1 + t2) / (sigma1 * sigma2)
        return torch.sqrt(torch.tensor(2.0)) * torch.arccosh(c)

    def __distance_beta(self,
                        point1: Tuple[torch.Tensor, torch.Tensor],
                        point2: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        alpha1 = point1[0]
        beta1 = point1[1]
        alpha2 = point2[0]
        beta2 = point2[1]

        t_vals = torch.linspace(0, 1, 100)
        gamma = torch.stack([
            alpha1 + (alpha2 - alpha1) * t_vals,
            beta1 + (beta2 - beta1) * t_vals
        ], dim=1)

        dists = []
        for i in range(len(t_vals) - 1):
            g1 = gamma[i]
            g2 = gamma[i + 1]
            delta = g2 - g1
            G = self.metric_matrix(g1)
            d = torch.sqrt(delta @ G @ delta)
            dists.append(d)

        return torch.sum(torch.stack(dists))