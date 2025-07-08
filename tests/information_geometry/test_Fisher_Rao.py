import unittest
import logging

# Force Geomstats to use Pytorch as a backend
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from typing import Tuple
from geomstats.information_geometry.normal import UnivariateNormalDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.poisson import PoissonDistributions
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.gamma import GammaDistributions
from geomstats.information_geometry.binomial import BinomialDistributions
from geomstats.information_geometry.base import InformationManifoldMixin
from information_geometry.fisher_rao import FisherRao
import torch
import python
from python import SKIP_REASON


class FisherRaoTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        fisher_rao = FisherRao(exponential_distributions, (1.0, 2.0))
        logging.info(f'Fisher-Rao:\n{str(fisher_rao)}')
        self.assertTrue(fisher_rao.fisher_rao_metric.signature == (1, 0))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_samples(self):
        FisherRaoTest.__sample_distribution(ExponentialDistributions(equip=True), (-2.0, 2.0), 8)
        FisherRaoTest.__sample_distribution(GeometricDistributions(equip=True), (1.0, 12.0), 8)
        FisherRaoTest.__sample_distribution(PoissonDistributions(equip=True), (0.0, 20.0), 6)
        FisherRaoTest.__sample_distribution(BinomialDistributions(equip=True, n_draws=10), (0.0, 20.0), 6)
        self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_distances(self):
        FisherRaoTest.__distance(ExponentialDistributions(equip=True), (-2.0, 2.0))
        FisherRaoTest.__distance(GeometricDistributions(equip=True), (1.0, 12.0))
        FisherRaoTest.__distance(PoissonDistributions(equip=True), (0.0, 20.0))
        FisherRaoTest.__distance(BinomialDistributions(equip=True, n_draws=8), (0.0, 20.0))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inner_products_1(self):
        v = torch.Tensor([1.0])
        w = torch.Tensor([-1.0])
        FisherRaoTest.__inner_product(ExponentialDistributions(equip=True), (-2.0, 2.0), v, w)
        FisherRaoTest.__inner_product(GeometricDistributions(equip=True), (1.0, 12.0), v, w)
        FisherRaoTest.__inner_product(PoissonDistributions(equip=True), (0.0, 20.0), v, w)
        FisherRaoTest.__inner_product(BinomialDistributions(equip=True, n_draws=10), (0.0, 20.0), v, w)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inner_products_2(self):
        v = torch.Tensor([2.0])
        w = torch.Tensor([0.5])
        FisherRaoTest.__inner_product(ExponentialDistributions(equip=True), (-2.0, 2.0), v, w)
        FisherRaoTest.__inner_product(GeometricDistributions(equip=True), (1.0, 12.0), v, w)
        FisherRaoTest.__inner_product(PoissonDistributions(equip=True), (0.0, 20.0), v, w)
        FisherRaoTest.__inner_product(BinomialDistributions(equip=True, n_draws=10), (0.0, 20.0), v, w)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inner_products_3(self):
        v = torch.Tensor([1.0])
        w = torch.Tensor([0.0])
        FisherRaoTest.__inner_product(ExponentialDistributions(equip=True), (-2.0, 2.0), v, w)
        FisherRaoTest.__inner_product(GeometricDistributions(equip=True), (1.0, 12.0), v, w)
        FisherRaoTest.__inner_product(PoissonDistributions(equip=True), (0.0, 20.0), v, w)
        FisherRaoTest.__inner_product(BinomialDistributions(equip=True, n_draws=10), (0.0, 20.0), v, w)

    def test_inner_norm_1(self):
        v = torch.Tensor([0.5])
        FisherRaoTest.__norm(ExponentialDistributions(equip=True), (-2.0, 2.0), v)
        FisherRaoTest.__norm(GeometricDistributions(equip=True), (1.0, 12.0), v)
        FisherRaoTest.__norm(PoissonDistributions(equip=True), (0.0, 20.0), v)
        FisherRaoTest.__norm(BinomialDistributions(equip=True, n_draws=10), (0.0, 20.0), v)

    def test_inner_norm_2(self):
        v = torch.Tensor([1.0])
        FisherRaoTest.__norm(ExponentialDistributions(equip=True), (-2.0, 2.0), v)
        FisherRaoTest.__norm(GeometricDistributions(equip=True), (1.0, 12.0), v)
        FisherRaoTest.__norm(PoissonDistributions(equip=True), (0.0, 20.0), v)
        FisherRaoTest.__norm(BinomialDistributions(equip=True, n_draws=10), (0.0, 20.0), v)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_gamma_samples(self):
        gamma_distributions = GammaDistributions(equip=True)
        fisher_rao = FisherRao(gamma_distributions, (-2.0, 2.0))
        logging.info(str(fisher_rao))
        samples = fisher_rao.samples(n_samples=4)
        logging.info('\n'.join([str(x) for x in samples]))
        self.assertTrue(fisher_rao.belongs(samples))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_metric_matrix_exponential(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        fisher_rao = FisherRao(exponential_distributions, (1.0, 2.0))
        metric = fisher_rao.metric_matrix()
        logging.info(f'Fisher-Rao metric for exponential: {metric}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_metric_matrix_gamma(self):
        gamma_distributions = GammaDistributions(equip=True)
        fisher_rao = FisherRao(gamma_distributions, (1.0, 2.0))
        metric = fisher_rao.metric_matrix()
        logging.info(f'Fisher-Rao metric for exponential: {metric}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_distance_exponential(self):
        import torch

        low_bound = 0.0
        upper_bound = 10.0
        fisher_rao = FisherRao(ExponentialDistributions(equip=True), bounds=(low_bound, upper_bound))
        values = fisher_rao.samples(2)
        metrics = [fisher_rao.metric_matrix(x) for x in values]
        logging.info(f'Exponential Metrics:\n{metrics[0]}, {metrics[1]}')
        inputs = [torch.Tensor(x) for x in values]
        distance = fisher_rao.distance(inputs[0], inputs[1])
        logging.info(f'Exponential Distance {distance}')
        fisher_rao.visualize_diff(values[0], values[1], r"$\theta$")

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_exponentials(self):
        import torch

        low_bound = 0.0
        upper_bound = 2.0
        fisher_rao = FisherRao(ExponentialDistributions(equip=True), bounds =(low_bound, upper_bound))
        values = fisher_rao.samples(128)
        min_theta = f'{float(torch.min(values)):.4f}'
        max_theta = f'{float(torch.max(values)):.4f}'
        fisher_rao.visualize_pdfs(values,
                                  rf"Exp. Distribution Manifold $\theta$ 128 samples [{min_theta}, {max_theta}]")

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_normal_mu(self):
        import torch

        low_bound = 0.0
        upper_bound = 2.0
        fisher_rao = FisherRao(UnivariateNormalDistributions(equip=True), bounds=(low_bound, upper_bound))
        values = fisher_rao.samples(48)
        min_mu = f'{float(torch.min(values[:, 0])):.4f}'
        max_mu = f'{float(torch.max(values[:, 0])):.4f}'
        fisher_rao.visualize_pdfs(values,
                                  rf"Normal Distribution Manifold $\mu$ 48 samples [{min_mu}, {max_mu}]")
        min_sigma = f'{float(torch.min(values[:, 1])):.4f}'
        max_sigma = f'{float(torch.max(values[:, 1])):.4f}'
        fisher_rao.visualize_pdfs(values,
                                  rf"Normal Distribution Manifold  $\sigma$ 48 samples [{min_sigma}, {max_sigma}]")

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_distance_univariate_normal(self):
        import torch
        fisher_rao = FisherRao(UnivariateNormalDistributions(equip=True), (1.0, 2.0))
        # values = fisher_rao.samples(2)
        # metrics = [fisher_rao.metric_matrix(x) for x in values]
        metrics = fisher_rao.metric_matrix()
        logging.info(f'Univariate Normal Metrics:\n{metrics[0]}, {metrics[1]}')
        inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        distance = fisher_rao.distance(inputs[0], inputs[1])
        logging.info(f'Univariate Normal  Distance  {distance}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_distance_geometric(self):
        import torch
        fisher_rao = FisherRao(GeometricDistributions(equip=True), (1.0, 2.0))
        # values = fisher_rao.samples(2)
        # metrics = [fisher_rao.metric_matrix(x) for x in values]
        metrics = fisher_rao.metric_matrix()
        logging.info(f'Geometric Metrics:\n{metrics[0]}')
        inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        distance = fisher_rao.distance(inputs[0], inputs[1])
        logging.info(f'Geometric Distance  {distance}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inner_product(self):
        import torch

        fisher_rao = FisherRao(GeometricDistributions(equip=True), (1.0, 2.0))
        # inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inputs = fisher_rao.samples(2)
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([1.0]), torch.Tensor([0.2]))
        logging.info(f'Geometric inner product  {inner_product}')

        fisher_rao = FisherRao(BetaDistributions(equip=True), (0.2, 0.8))
        inputs = fisher_rao.samples(2)
        # inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([0.5, 0.8]), torch.Tensor([0.2, 0.6]))
        logging.info(f'Beta inner product  {inner_product}')

        fisher_rao = FisherRao(UnivariateNormalDistributions(equip=True), (0.0, 1.0))
        # inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inputs = fisher_rao.samples(2)
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([0.5, 0.8]), torch.Tensor([0.2, 0.6]))
        logging.info(f'Univariate Normal inner product  {inner_product}')

        fisher_rao = FisherRao(GeometricDistributions(equip=True), (1.0, 2.0))
        inputs = fisher_rao.samples(2)
        # inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([1.0]), torch.Tensor([0.2]))
        logging.info(f'Exponential inner product  {inner_product}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_distance_beta(self):
        import torch
        fisher_rao = FisherRao(BetaDistributions(equip=False), (0.2, 0.5))
        values = fisher_rao.samples(6)
        # metrics = [fisher_rao.metric_matrix(x) for x in values[4:6]]
        metrics = fisher_rao.metric_matrix()
        logging.info(f'Beta Metrics:\n{metrics[0]}, {metrics[1]}')
        inputs = [torch.Tensor(x) for x in values]
        distance = fisher_rao.distance(inputs[0], inputs[1])
        logging.info(f'Beta  Distance  {distance}')

    @staticmethod
    def __sample_distribution(stats_manifold: InformationManifoldMixin,
                              bounds: Tuple[float, float],
                              n_samples: int) -> None:
        fisher_rao = FisherRao(stats_manifold, bounds)
        random_samples = fisher_rao.samples(n_samples=n_samples)
        assert fisher_rao.belongs(random_samples)

        results = ', '.join([f'{(float(x)):.4f}' for x in random_samples])
        logging.info(f"{stats_manifold.__class__.__name__} samples:\n{results}")

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    @staticmethod
    def __distance(stats_manifold: InformationManifoldMixin, bounds: Tuple[float, float]):
        fisher_rao = FisherRao(stats_manifold, bounds=bounds)
        logging.info(stats_manifold.__class__.__name__)
        values = fisher_rao.samples(2)
        distance = fisher_rao.distance(values[0], values[1])
        logging.info(f'd({float(values[0]):.4f}, {float(values[1]):.4f}) = {float(distance):.4f}')
        fisher_rao.visualize_diff(values[0], values[1], r"$\theta$")

    @staticmethod
    def __inner_product(stats_manifold: InformationManifoldMixin,
                        bounds: Tuple[float, float],
                        v: torch.Tensor,
                        w: torch.Tensor):
        fisher_rao = FisherRao(stats_manifold, bounds)
        point = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(point[0], v, w)
        logging.info(f'{stats_manifold.__class__.__name__} {float(v)} dot {float(w)} = {float(inner_product):.3f}')

    @staticmethod
    def __norm(stats_manifold: InformationManifoldMixin, bounds: Tuple[float, float],v: torch.Tensor):
        fisher_rao = FisherRao(stats_manifold, bounds)
        point = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        norm = fisher_rao.norm(point[0], v)
        logging.info(f'{stats_manifold.__class__.__name__} Norm({float(v)}) = {float(norm):.3f}')