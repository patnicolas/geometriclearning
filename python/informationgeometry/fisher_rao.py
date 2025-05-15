__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from typing import Tuple, AnyStr, List
import numpy as np
import torch

from geometry import GeometricException
ParamType = torch.Tensor | Tuple[torch.Tensor, torch.Tensor]


class FisherRao(object):
    """
    Class that wraps the computation of the Riemannian metric for some common statistical manifolds and support the
    computation of:
    - Riemannian metric (Fisher_Rao matrix) [Using Geomstats]
    - Inner product of distribution parameters on tangent space  [Homegrown implementation]
    - Distance between two distributions parameters [Homegrown implementation]

    The Riemannan metric for a distribution on a manifold is defined by
    .. math::
           g_{j k}(\theta)=\int_X \frac{\partial \log p(x, \theta)}
                {\partial \theta_j}\frac{\partial \log p(x, \theta)}
                {\partial \theta_k} p(x, \theta) d x

    The metric matrix for a distribution with pdf f, and parmeters \theta is defined as
    .. math::
                  I_{ij} = \int \
                        \partial_{i} f_{\theta}(x)\
                        \partial_{j} f_{\theta}(x)\
                        \frac{1}{f_{\theta}(x)}

    The inner-product of two vectors or tensors in the tangent space at a given parameter point on the statistical
    manifold is defined as
    .. math::
                  \partial_k I_{ij} = \int\
                        \partial_{ki}^2 f\partial_j f \frac{1}{f} + \
                        \partial_{kj}^2 f\partial_i f \frac{1}{f} - \
                        \partial_i f \partial_j f \partial_k f \frac{1}{f^2}

    """
    # List of statistical manifolds supported by this class
    valid_info_manifolds = [
        'ExponentialDistributions', 'BetaDistributions', 'GammaDistributions', 'UnivariateNormalDistributions',
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
        assert class_name in FisherRao.valid_info_manifolds, f'Information Geometry for {class_name} is not supported'

        self.info_manifold = info_manifold
        self.fisher_rao_metric = FisherRaoMetric(info_manifold, bounds)

    def __str__(self) -> AnyStr:
        return (f'Information Manifold: {self.info_manifold.__class__.__name__}'
                f'\nFisher-Rao metric:\n{self.fisher_rao_metric.signature}')

    def belongs(self, points: List[np.array]) -> bool:
        """
        Test if a list of points belongs to this statistical manifold
        @param points: Points on the statistical manifold
        @type points: List of Numpy arrays
        @return: True if each point belongs to the manifold, False if one or more points do not belong to
        the manifold
        @rtype: bool
        """
        all_pts_belongs = [self.info_manifold.belongs(pt) for pt in points]
        return all(all_pts_belongs)

    def samples(self, n_samples: int) -> np.array:
        return self.info_manifold.random_point(n_samples)

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

    def distance(self, point1: ParamType, point2: ParamType) -> torch.Tensor:
        """
        Compute the distance between two distributions on the manifold given parameters theta1
        and theta2.
        The types of theta1 and theta2 are torch.Tensor for single parameters distribution (i.e.
        exponential) or a Tuple for multi-parameters distribution (i.e. Beta)
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
            case 'BetaDistributions':
                return self.__distance_beta(point1, point2)
            case 'GeometricDistributions':
                return self.__distance_geometric(point1, point2)
            case 'UnivariateNormalDistributions':
                return FisherRao.__distance_univariate_normal(point1, point2)
            case _:
                raise GeometricException(f'Distance for {self.info_manifold.__class__.__name__} not supported')


    def inner_product(self, point: torch.Tensor, v: torch.Tensor, w: torch.Tensor)-> torch.Tensor | None:
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return v*w / (point ** 2)
            case 'BetaDistributions':
                g = self.metric_matrix(point)
                return v @ g @ w
            case 'GeometricDistributions':
                g = 1 / point ** 2 + 1 / (1 - point) ** 2
                return g*v*w
            case 'UnivariateNormalDistributions':
                sigma_sq_inv = 1.0 / (point[1] ** 2)
                return sigma_sq_inv * v[0] * w[0] + 2 * sigma_sq_inv * v[1] * w[1]
            case _:
                raise GeometricException(f'inner product for {self.info_manifold.__class__.__name__} not supported')

    def visualize_pdf(self, parameters1: torch.Tensor,  parameters2: torch.Tensor, label: AnyStr) -> None:
        support = self.fisher_rao_metric.support
        x = np.linspace(support[0], support[1], 100)
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
        support = self.fisher_rao_metric.support
        x = np.linspace(support[0], support[1], 100)
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

    def __distance_geometric(self, point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor | None:
        from scipy.integrate import quad

        p1 = point1.numpy()
        p2 = point2.numpy()
        def _metric(p: float) -> float:
            return np.sqrt(1/p**2 + 1/(1-p)**2)

        distance, _ = quad(_metric, p1 ,p2)
        return abs(distance)