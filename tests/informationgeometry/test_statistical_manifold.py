import unittest
import logging

import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from geomstats.information_geometry.normal import UnivariateNormalDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.gamma import GammaDistributions
from informationgeometry.statistical_manifold import StatisticalManifold


class StatisticalManifoldTest(unittest.TestCase):

    @unittest.skip('ignore')
    def test_init(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        statistical_manifold = StatisticalManifold(exponential_distributions, (1.0, 2.0))
        logging.info(str(statistical_manifold))
        self.assertTrue(statistical_manifold.fisher_rao_metric.signature == (1, 0))

    @unittest.skip('ignore')
    def test_exponential_samples(self):
        exponential_distributions = ExponentialDistributions(equip=True)
        statistical_manifold = StatisticalManifold(exponential_distributions, (-2.0, 2.0))
        samples = statistical_manifold.samples(n_samples=4)
        logging.info('\n'.join([str(x) for x in samples]))


