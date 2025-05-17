__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from typing import Tuple, AnyStr, List
import numpy as np
import torch


class StatisticalManifold(object):
    """
    Class that wraps the basic implementation of the most common statistical, Riemannian manifolds (distribution)
    using the Fisher-Rao metric tensor.
    The fisher-Rao metric related method such as inner product, distance or norm are implemented in the subclass
    FisherRao
    The key methods supported by this class are
    - sample and randomly generated point on a given statistical manifold
    - exponential map
    - logarithm map
    """
    valid_info_manifolds = [
        'ExponentialDistributions', 'BetaDistributions', 'PoissonDistributions', 'UnivariateNormalDistributions',
        'GeometricDistributions'
    ]

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
        class_name = info_manifold.__class__.__name__
        assert class_name in StatisticalManifold.valid_info_manifolds, f'Information Geometry for {class_name} is not supported'

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
                f'\nFisher-Rao metric:\n{self.fisher_rao_metric.signature}')

    def belongs(self, points: List[torch.Tensor]) -> bool:
        """
        Test if a list of points belongs to this statistical manifold
        @param points: Points on the statistical manifold
        @type points: List of Numpy arrays
        @return: True if each point belongs to the manifold, False if one or more points do not belong to
        the manifold
        @rtype: bool
        """
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
        return torch.Tensor(self.info_manifold.random_point(n_samples))

    def metric_matrix(self, base_point: np.array = None) -> np.array:
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
        return self.fisher_rao_metric.metric_matrix(base_point)

    def exp(self, tangent_vec: torch.Tensor, base_point:  torch.Tensor) -> torch.Tensor:
        """
        Define the exponential map for the default metric for a given family of probability density functions. This
        method implements the computation of the end point on the geodesic given a point on the manifold and
        a tangent vector.
        Given a geodesic G and a base point p. the exponential map computes the end point as
        .. math::
             G_{v}(0)=p \ \ \ \ ; \bigtriangledown _{v}\left ( G_{v} \right )(0)=v \ \ ; \ \ \ exp_{p}(v)=G_{v}(1)

        @param tangent_vec: Tangent vector for the statistical manifold of probability distribution parameters
        @type tangent_vec: torch.Tensor
        @param base_point: Point on the statistical manifold
        @type base_point: torch.Tensor
        @return: Value of exp(v) or end point on the manifold
        @rtype: torch.Tensor
        """
        return torch.Tensor(self.fisher_rao_metric.exp(tangent_vec.numpy(), base_point.numpy()))

    def log(self, manifold_point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
        """
        Implements the logarithm map to compute the tangent vector given a base point and a point of manifold.
        Given two points theta1 and theta2 on the manifold, the logarithm map compute the vector v
        .. math::
            log_{\theta_{1)(\theta_{2) = v

        @param manifold_point: Second point on the manifold, along a geodesic.
        @type manifold_point: tprch.Tensor
        @param base_point: Base point on the manifold
        @type base_point:tprch.Tensor
        @return: Tangent vectpr v
        @rtype: tprch.Tensor
        """
        return torch.Tensor(self.fisher_rao_metric.log(manifold_point.numpy(), base_point.numpy()))

    """ ----------------------------  Visualization methods ----------------------   """

    def visualize_pdf(self, parameters1: torch.Tensor,  parameters2: torch.Tensor, label: AnyStr) -> None:
        x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
        pdf_1 = self.info_manifold.point_to_pdf(parameters1)
        pdf_2 = self.info_manifold.point_to_pdf(parameters2)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.plot(x, pdf_1(x), label=f'{label} {float(parameters1):.4f}', linewidth=2)
        plt.plot(x, pdf_2(x), label=f'{label}  {float(parameters2):.4f}', linewidth=2, linestyle='--')
        plt.plot(x, pdf_1(x) - pdf_2(x), label=f'Diff {label} {(float(parameters1)-float(parameters2)):.4f}', linewidth=2)
        plt.xlabel('x', fontdict={'fontsize': 16})
        plt.ylabel('pdf', fontdict={'fontsize': 16})
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_pdfs(self, parameters: List[torch.Tensor], param_label: AnyStr) -> None:
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

