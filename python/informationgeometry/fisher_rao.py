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
    valid_info_manifolds = [
        'ExponentialDistributions', 'PoissonDistributions', 'BinomialDistribution', 'DirichletDistributions',
        'BetaDistributions', 'GammaDistributions', 'MultinomialDistributions', 'UnivariateNormalDistributions',
        'CenteredNormalDistributions', 'DiagonalNormalDistributions', 'GeneralNormalDistributions', 'GeometricDistributions'
    ]

    def __init__(self, info_manifold: InformationManifoldMixin, bounds: Tuple[float, float]) -> None:
        class_name = info_manifold.__class__.__name__
        assert class_name in FisherRao.valid_info_manifolds, \
            f'Information Geometry for {info_manifold.__class__.__name__} is not supported'

        self.info_manifold = info_manifold
        self.fisher_rao_metric = FisherRaoMetric(info_manifold, bounds)

    def __str__(self) -> AnyStr:
        return (f'Information Manifold: {self.info_manifold.__class__.__name__}'
                f'\nFisher-Rao metric:\n{self.fisher_rao_metric.signature}')

    def belongs(self, points: List[np.array]) -> bool:
        all_pts_belongs = [self.info_manifold.belongs(pt) for pt in points]
        return all(all_pts_belongs)

    def samples(self, n_samples: int) -> np.array:
        return self.info_manifold.random_point(n_samples)

    def metric_matrix(self, base_point: np.array = None) -> np.array:
        base_point = self.info_manifold.random_point(1) if base_point is None else base_point
        assert self.info_manifold.belongs(base_point)
        metric = self.fisher_rao_metric.metric_matrix(base_point)
        return metric

    def distance(self, theta1: ParamType, theta2: ParamType) -> torch.Tensor | None:
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return torch.abs(torch.log(theta2) - torch.log(theta1))
            case 'BetaDistributions':
                return self.__distance_beta(theta1, theta2)
            case 'GeometricDistributions':
                return self.__distance_geometric(theta1, theta2)
            case 'UnivariateNormalDistributions':
                return FisherRao.__distance_univariate_normal(theta1, theta2)
            case _:
                raise GeometricException(f'Distance for {self.info_manifold.__class__.__name__} not supported')


    def inner_product(self, theta: torch.Tensor, v: torch.Tensor, w: torch.Tensor)-> torch.Tensor | None:
        match self.info_manifold.__class__.__name__:
            case 'ExponentialDistributions':
                return v*w / (theta ** 2)
            case 'BetaDistributions':
                g = self.metric_matrix(theta)
                return v @ g @ w
            case 'GeometricDistributions':
                g = 1/theta**2 + 1/(1-theta)**2
                return g*v*w
            case 'UnivariateNormalDistributions':
                sigma_sq_inv = 1.0 / (theta[1] ** 2)
                return sigma_sq_inv * v[0] * w[0] + 2 * sigma_sq_inv * v[1] * w[1]
            case _:
                raise GeometricException(f'inner product for {self.info_manifold.__class__.__name__} not supported')

    def visualize_pdf(self, parameters1: torch.Tensor,  parameters2: torch.Tensor) -> None:
        support = self.fisher_rao_metric.support
        x = np.linspace(support[0], support[1], 100)
        pdf_1 = self.info_manifold.point_to_pdf(parameters1)
        pdf_2 = self.info_manifold.point_to_pdf(parameters2)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        distribution_cat = self.info_manifold.__class__.__name__
        plt.plot(x, pdf_1(x), label=f'{distribution_cat} {float(parameters1):.4f}', linewidth=2)
        plt.plot(x, pdf_2(x), label=f'{distribution_cat}  {float(parameters2):.4f}', linewidth=2, linestyle='--')
        plt.plot(x, pdf_1(x) - pdf_2(x), label=f'Difference {(float(parameters1)-float(parameters2)):.4f}', linewidth=2)
        plt.xlabel('x', fontdict={'fontsize': 16})
        plt.ylabel('pdf', fontdict={'fontsize': 16})
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def __distance_univariate_normal(theta1: Tuple[torch.Tensor, torch.Tensor],
                                     theta2: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        mu1 = theta1[0]
        sigma1 = theta1[1]
        mu2 = theta2[0]
        sigma2 = theta2[1]
        if sigma1 < 1e-12 or sigma2 < 1e-12:
            return None
        t1 = (mu1 - mu2) ** 2  # (mu1 - mu2)**2
        t2 = (sigma1 - sigma2) ** 2  # (sigma1 - sigma2)**2
        c = 1 + 0.5 * (t1 + t2) / (sigma1 * sigma2)
        return torch.sqrt(torch.tensor(2.0)) * torch.arccosh(c)

    def __distance_beta(self,
                        theta1: Tuple[torch.Tensor, torch.Tensor],
                        theta2: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        alpha1 = theta1[0]
        beta1 = theta1[1]
        alpha2 = theta2[0]
        beta2 = theta2[1]

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

    def __distance_geometric(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor | None:
        from scipy.integrate import quad

        p1 = theta1.numpy()
        p2 = theta2.numpy()

        def _metric(p: float) -> float:
            return np.sqrt(1/p**2 + 1/(1-p)**2)

        distance, _ = quad(_metric, p1 ,p2)
        return abs(distance)