import unittest
import logging

import torch
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from geomstats.information_geometry.binomial import BinomialDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.poisson import PoissonDistributions
from informationgeometry.statistical_manifold import StatisticalManifold
from typing import Tuple, AnyStr



class StatisticalManifoldTest(unittest.TestCase):

    @unittest.skip('ignore')
    def test_init(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        statistical_manifold = StatisticalManifold(exponential_distributions, (1.0, 2.0))
        logging.info(f'\n{statistical_manifold=}')
        self.assertTrue(statistical_manifold.fisher_rao_metric.signature == (1, 0))

    @unittest.skip('ignore')
    def test_exponential_samples(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        statistical_manifold = StatisticalManifold(exponential_distributions, (-2.0, 2.0))
        samples = statistical_manifold.samples(n_samples=4)
        logging.info('\n'.join([str(x) for x in samples]))

    @unittest.skip('ignore')
    def test_metric_matrix_exponential(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        statistical_manifold = StatisticalManifold(exponential_distributions, (1.0, 2.0))
        metric = statistical_manifold.metric_matrix()
        logging.info(f'Fisher-Rao metric for exponential: {metric}')

    @unittest.skip('ignore')
    def test_metric_matrix_gamma(self):
        gamma_distributions = GammaDistributions(equip=True)
        statistical_manifold = StatisticalManifold(gamma_distributions, (1.0, 2.0))
        metric = statistical_manifold.metric_matrix()
        logging.info(f'Fisher-Rao metric for exponential: {metric}')

    @unittest.skip('ignore')
    def test_exp_map_exponential(self):
        exponential_distributions = ExponentialDistributions(equip=False)
        statistical_manifold = StatisticalManifold(exponential_distributions, (-2.0, 2.0))
        tgt_vector = torch.Tensor([0.5])
        base_point = statistical_manifold.samples(1)
        end_point = statistical_manifold.exp(tgt_vector, base_point)
        logging.info(end_point)

    @unittest.skip('ignore')
    def test_exp_map_geometric(self):
        exponential_distributions = GeometricDistributions(equip=True)
        statistical_manifold = StatisticalManifold(exponential_distributions, (1, 10))
        tgt_vector = torch.Tensor([0.5])
        base_point = statistical_manifold.samples(1)
        end_point = statistical_manifold.exp(tgt_vector, base_point)
        logging.info(end_point)

    @unittest.skip('ignore')
    def test_visualize_exponentials(self):
        StatisticalManifoldTest.visualize(distribution=ExponentialDistributions(equip=True),
                                          bounds=(0.0, 2.0),
                                          num_samples=128,
                                          param_desc="$ \\theta \in$")
        self.assertTrue(True)


    @unittest.skip('ignore')
    def test_visualize_poissons(self):
        StatisticalManifoldTest.visualize(distribution=PoissonDistributions(equip=True),
                                          bounds=(0.0, 20.0),
                                          num_samples=128,
                                          param_desc='$\\lambda \in$')
        self.assertTrue(True)

    @unittest.skip('ignore')
    def test_visualize_binomial(self):
        n_draws = 40
        StatisticalManifoldTest.visualize(distribution=BinomialDistributions(equip=True, n_draws=n_draws),
                                          bounds=(0.0, n_draws/2),
                                          num_samples=128,
                                          param_desc='n_draws=40,  p $ \in$')
        self.assertTrue(True)


    # @unittest.skip('ignore')
    def test_visualize_geometric(self):
        StatisticalManifoldTest.visualize(distribution=GeometricDistributions(equip=True),
                                          bounds=(1.0, 10.0),
                                          num_samples=128,
                                          param_desc='p $ \in$')
        self.assertTrue(True)



    @staticmethod
    def visualize(distribution: InformationManifoldMixin,
                  bounds: Tuple[float, float],
                  num_samples: int,
                  param_desc: AnyStr) -> None:
        import torch

        statistical_manifold = StatisticalManifold(distribution, bounds=bounds)
        values = statistical_manifold.samples(num_samples)
        min_theta = f'{float(torch.min(values)):.4f}'
        max_theta = f'{float(torch.max(values)):.4f}'
        statistical_manifold.visualize_pdfs(
            values,
            rf"{distribution.__class__.__name__} {num_samples} samples {param_desc} [{min_theta}, {max_theta}]"
        )
