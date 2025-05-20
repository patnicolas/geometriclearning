import unittest
import logging

# Force Geomstats to use Pytorch as a backend
import os

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from geomstats.information_geometry.normal import UnivariateNormalDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.gamma import GammaDistributions
from informationgeometry.fisher_rao import FisherRao


class FisherRaoTest(unittest.TestCase):

    @unittest.skip('ignore')
    def test_init(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        fisher_rao = FisherRao(exponential_distributions, (1.0, 2.0))
        logging.info(str(fisher_rao))
        self.assertTrue(fisher_rao.fisher_rao_metric.signature == (1, 0))

    @unittest.skip('ignore')
    def test_exponential_samples(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        fisher_rao = FisherRao(exponential_distributions, (-2.0, 2.0))
        samples = fisher_rao.samples(n_samples=4)
        logging.info('\n'.join([str(x) for x in samples]))

    @unittest.skip('ignore')
    def test_gamma_samples(self):
        gamma_distributions = GammaDistributions(equip=True)
        fisher_rao = FisherRao(gamma_distributions, (-2.0, 2.0))
        logging.info(str(fisher_rao))
        samples = fisher_rao.samples(n_samples=4)
        logging.info('\n'.join([str(x) for x in samples]))
        self.assertTrue(fisher_rao.belongs(samples))

    @unittest.skip('ignore')
    def test_metric_matrix_exponential(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        fisher_rao = FisherRao(exponential_distributions, (1.0, 2.0))
        metric = fisher_rao.metric_matrix()
        logging.info(f'Fisher-Rao metric for exponential: {metric}')

    @unittest.skip('ignore')
    def test_metric_matrix_gamma(self):
        gamma_distributions = GammaDistributions(equip=True)
        fisher_rao = FisherRao(gamma_distributions, (1.0, 2.0))
        metric = fisher_rao.metric_matrix()
        logging.info(f'Fisher-Rao metric for exponential: {metric}')

    @unittest.skip('ignore')
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

    @unittest.skip('ignore')
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


    @unittest.skip('ignore')
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

    @unittest.skip('ignore')
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

    # @unittest.skip('ignore')
    def test_inner_product(self):
        import torch

        fisher_rao = FisherRao(GeometricDistributions(equip=True), (1.0, 2.0))
        inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([1.0]), torch.Tensor([0.2]))
        logging.info(f'Geometric inner product  {inner_product}')

        fisher_rao = FisherRao(BetaDistributions(equip=True), (0.2, 0.8))
        inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([0.5, 0.8]), torch.Tensor([0.2, 0.6]))
        logging.info(f'Beta inner product  {inner_product}')

        fisher_rao = FisherRao(UnivariateNormalDistributions(equip=True), (0.0, 1.0))
        inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([0.5, 0.8]), torch.Tensor([0.2, 0.6]))
        logging.info(f'Univariate Normal inner product  {inner_product}')

        fisher_rao = FisherRao(GeometricDistributions(equip=True), (1.0, 2.0))
        inputs = [torch.Tensor(x) for x in fisher_rao.samples(2)]
        inner_product = fisher_rao.inner_product(inputs[0], torch.Tensor([1.0]), torch.Tensor([0.2]))
        logging.info(f'Exponential inner product  {inner_product}')

    @unittest.skip('ignore')
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


