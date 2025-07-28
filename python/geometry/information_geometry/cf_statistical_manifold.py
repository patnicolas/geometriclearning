__author__ = "Patrick R. Nicolas"
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
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from typing import Tuple, AnyStr, List
import numpy as np
import torch
from geometry import GeometricException
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
__all__ = ['CFStatisticalManifold']


class CFStatisticalManifold(object):
    """
    Class that wraps the basic implementation of the most common statistical, Riemannian manifolds (distribution)
    using the Fisher-Rao metric tensor.
    The fisher-Rao metric related method such as inner product, distance or norm are implemented in the subclass
    FisherRao
    The key methods supported by this class are
    - sample and randomly generated point on a given statistical manifold
    - exponential map
    - logarithm map

    NOTES:
        - This class supports only the closed-form distributions that are available through Geomstats API
        - Although other classes relies on Numpy binding for back-end computations, Statistical manifolds reqyire
          PyTorch be selected as back-end

    REFERENCE:
    """
    closed_form_manifolds = [
        'ExponentialDistributions', 'PoissonDistributions', 'BinomialDistributions', 'GeometricDistributions'
    ]

    def __init__(self, info_manifold: InformationManifoldMixin, bounds: Tuple[float, float]) -> None:
        """
        Constructor for the computation of the Fisher_Rao metric for the most common family of uni-variate and
        bivariate distributions
        @param info_manifold: Information manifold or Class of distribution
        @type info_manifold: A subclass of InformationManifoldMixin such as ExponentialDistributions or
        GeometricDistributions
        @param bounds: Tuple of values which set the bounds of input to probability density function
        @type bounds: Tuple[float, float]
        """
        class_name = info_manifold.__class__.__name__
        assert class_name in CFStatisticalManifold.closed_form_manifolds, \
            f'Information Geometry for {class_name} is not supported'

        self.info_manifold = info_manifold
        self.fisher_rao_metric = FisherRaoMetric(info_manifold, bounds)

    def get_bounds(self) -> Tuple[float, float]:
        """
        Retrieve the bounds or support for the metric associated with this statistical manifold
        @return: Tuple of upper and lower bound
        @rtype: Tuple[float, float]
        """
        return self.fisher_rao_metric.support

    def __str__(self) -> AnyStr:
        return (f'Information Manifold: {self.info_manifold.__class__.__name__}'
                f'Fisher-Rao Metric Signature: {self.fisher_rao_metric.signature}')

    def belongs(self, points: List[torch.Tensor]) -> bool:
        """
        Test if a list of points belongs to this statistical manifold
        @param points: Points on the statistical manifold
        @type points: List of Numpy arrays
        @return: True if each point belongs to the manifold, False if one or more points do not belong to
        the manifold
        @rtype: bool
        """
        assert len(points) > 0, 'Cannot test undefined number of points belongs to a manifold'

        all_pts_belongs = [self.info_manifold.belongs(pt.numpy()) for pt in points]
        return all(all_pts_belongs)

    def samples(self, n_samples: int) -> torch.Tensor:
        """
        Generate random samples of point on the statistical manifold
        @param n_samples: Number of samples
        @type n_samples: int
        @return: Tensor of the random samplers
        @rtype: torch.Tensor
        """
        assert n_samples > 0, f'Number of samples {n_samples} should be > 0'

        lower_bound, upper_bound = self.get_bounds()
        delta = upper_bound - lower_bound \
            if self.info_manifold.__class__.__name__ == 'PoissonDistributions' else 1.0

        return torch.Tensor(self.info_manifold.random_point(n_samples))*delta

    def metric_matrix(self, base_point: torch.Tensor = None) -> torch.Tensor:
        """
        Computation of the Fisher_Rao metric at a given point. Contrary to the computation of
            the distance and inner product, this method invoke Geomstats API.
            If the point on the manifold is not provided, we select a random point
        @param base_point: Point of the manifold the metric is computed.
        @type base_point: Numpy array
        @return: Metric for this manifold
        @rtype: Numpy array
        """
        # Set the base point as a random point on the manifold if none is provided
        base_point = self.info_manifold.random_point(1) if base_point is None else base_point
        # Make sure the base point actually belongs to the manifold
        assert self.info_manifold.belongs(base_point)
        # Invoke the Geomstats method
        return torch.Tensor(self.fisher_rao_metric.metric_matrix(base_point))

    def exp(self, tangent_vec: torch.Tensor, base_point:  torch.Tensor) -> torch.Tensor:
        """
        Define the exponential map for the default metric for a given family of probability density functions. This
        method implements the computation of the end point on the geodesic given a point on the manifold and
        a tangent vector.
        Given a geodesic G and a base point p. the exponential map computes the end point as
        math::
             G_{v}(0)=p \ \ \ \ ; \bigtriangledown _{v}\left ( G_{v} \right )(0)=v \ \ ; \ \ \ exp_{p}(v)=G_{v}(1)

        @param tangent_vec: Tangent vector for the statistical manifold of probability distribution parameters
        @type tangent_vec: torch.Tensor
        @param base_point: Point on the statistical manifold
        @type base_point: torch.Tensor
        @return: Value of exp(v) or end point on the manifold
        @rtype: torch.Tensor
        """
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return base_point + tangent_vec
            case 'GeometricDistributions':
                phi_base_point = -2.0*torch.arctanh(torch.sqrt(1.0 - base_point))
                return 1.0 - torch.tanh(0.5*(phi_base_point+tangent_vec))**2
            case 'PoissonDistributions':
                return (torch.sqrt(base_point) + 0.5*tangent_vec)**2
            case 'BinomialDistributions':
                # Fixed the number of draws for analytical solution
                import math
                n_sqrt = math.sqrt(self.info_manifold.n_draws)
                return (torch.arcsin(torch.sqrt(base_point)) + 0.5 * tangent_vec/n_sqrt) ** 2
            case _:
                raise GeometricException(f'Exponential map for {self.info_manifold.__class__.__name__} not supported')

    def log(self, manifold_point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        """
        Implements the logarithm map to compute the tangent vector given a base point and a point of manifold.
        Given two points theta1 and theta2 on the manifold, the logarithm map compute the vector v
        math::
            log_{\theta_{1)(\theta_{2) = v

        @param manifold_point: Second point on the manifold, along a geodesic.
        @type manifold_point: torch.Tensor
        @param base_point: Base point on the manifold
        @type base_point: torch.Tensor
        @return: Tangent vector v
        @rtype: torch.Tensor
        """
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return base_point*torch.log(manifold_point/base_point)
            case 'GeometricDistributions':
                return -20*(torch.arctanh(torch.sqrt(1.0 - manifold_point)) -
                            torch.arctanh(torch.sqrt(1.0 - base_point)))
            case 'PoissonDistribution':
                return 2.0*(torch.sqrt(manifold_point) - torch.sqrt(base_point))
            case 'BinomialDistribution':
                # Fixed the number of draws for analytical solution
                n_sqrt = torch.sqrt(self.info_manifold.n_draws)
                return 2.0 * n_sqrt*(torch.arcsin(torch.sqrt(manifold_point)) - torch.arcsin(torch.sqrt(base_point)))
            case _:
                raise GeometricException(f'Logarithm map for {self.info_manifold.__class__.__name__} not supported')

    """ ----------------------------  Visualization methods -------------------------------------------   """

    def visualize_diff(self, parameter1: torch.Tensor, parameter2: torch.Tensor, param_label: AnyStr) -> None:
        """
        Visualize  (2D plot)  the difference or distance between two distribution with respective parameters
        parameter1, and parameter2. This method assumes a one-dimensional statistical manifolds

        @param parameter1: Parameter for the first closed-form manifold
        @type parameter1: Torch Tensor
        @param parameter2: Parameter for the second closed-form manifold
        @type parameter2: Torch Tensor
        @param param_label: Label that describe the type of parameter
        @type param_label: AnyStr
        """
        x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
        pdf_1 = self.info_manifold.point_to_pdf(parameter1)
        pdf_2 = self.info_manifold.point_to_pdf(parameter2)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.plot(x, pdf_1(x), label=f'{param_label} {float(parameter1):.4f}', linewidth=2)
        plt.plot(x, pdf_2(x), label=f'{param_label}  {float(parameter2):.4f}', linewidth=2, linestyle='--')
        plt.plot(x,
                 pdf_1(x) - pdf_2(x),
                 label=f'Diff {param_label} {(float(parameter1) - float(parameter2)):.4f}',
                 linewidth=2)
        plt.xlabel('x', fontdict={'fontsize': 16})
        plt.ylabel('pdf', fontdict={'fontsize': 16})
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(self.info_manifold.__class__.__name__)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_pdfs(self, parameters: torch.Tensor, param_label: AnyStr) -> None:
        """
        Visualize (2D plot) the probability densities for a given distribution with randomly selected parameters.
        @param parameters: Randomly sampled values of distribution parameter
        @type parameters: Torch Tensor
        @param param_label: Descriptor for the type of parameters to be shown on plot
        @type param_label: AnyStr
        """
        x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
        pdfs = [self.info_manifold.point_to_pdf(params) for params in parameters]

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 8))
        fig.set_facecolor('#F2F9FE')
        for idx, pdf in enumerate(pdfs):
            plt.plot(x, pdf(x), linewidth=2)

        plt.xlabel('x', fontdict={'fontsize': 16})
        plt.ylabel('pdf', fontdict={'fontsize': 16})
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(label=param_label, fontdict={'fontsize': 17})
        plt.legend(fontsize=10, ncol=2, loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def animate(self, parameters: torch.Tensor, param_label: AnyStr) -> None:
        from matplotlib.animation import FuncAnimation

        x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
        pdfs = [self.info_manifold.point_to_pdf(params) for params in parameters]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor('#F2F9FE')

        def update(frame: int) -> None:
            if frame < len(parameters):
                for idx in range(frame):
                        _x = x[0:frame]
                        ax.plot(_x, pdfs[idx](_x), linewidth=2)
                        if idx % 11 == 0:
                            title = rf"{self.info_manifold.__class__.__name__} {frame} samples {param_label}"
                            plt.title(label=title, fontdict={'fontsize': 16})
                            plt.grid(True)
                            plt.tight_layout()

        plt.xlabel('x', fontdict={'fontsize': 16})
        plt.ylabel('pdf', fontdict={'fontsize': 16})
        plt.tick_params(axis='both', which='major', labelsize=12)

        ani = FuncAnimation(fig, update, frames=len(parameters), interval=12, repeat=False, blit=False)
        # plt.show()
        ani.save('statistical_manifold_animation.mp4', writer='ffmpeg', fps=20, dpi=240)

