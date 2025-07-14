import unittest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import GradientDescent
from torch import Tensor

from geometry.frechet_estimator import FrechetEstimator
from geometry import InformationGeometricException

from geometry.visualization.hypersphere_plot import HyperspherePlot
from geometry.visualization.euclidean_plot import EuclideanPlot
from geometry.visualization.so3_plot import SO3Plot
import logging
import os
import python
from python import SKIP_REASON


class FrechetEstimatorTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_1(self):
        manifold = Hypersphere(dim=2, intrinsic=False, equip=True)
        weights = Tensor([0.4, 0.6, 1.0])
        X = Tensor([[0.4, -0.4, 0.6], [0.1, -1.0, 0.7], [0.7, -0.5, 0.0]])
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights)
            mean = frechet_estimator.estimate(X)
            logging.info(mean)
            self.assertTrue(False)
        except InformationGeometricException as e:
            logging.info(str(e))
            self.assertTrue(True)

    def test_estimate_hypersphere(self):
        manifold = Hypersphere(dim=2)
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights=None)
            np_points = frechet_estimator.rand(8)

            frechet_mean = frechet_estimator.estimate(np_points)
            hypersphere_plot = HyperspherePlot(np_points, frechet_mean)
            hypersphere_plot.show()

            euclidean_plot = EuclideanPlot(np_points, frechet_mean)
            euclidean_plot.show()
            euclidean_mean = FrechetEstimator.euclidean_mean(np_points)
            logging.info(f'\nFrechet mean:   {frechet_mean}\nEuclidean mean: {euclidean_mean}')

            self.assertTrue(True)
        except InformationGeometricException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_estimate_Euclidean(self):
        manifold = Hypersphere(dim=2)
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights=None)
            np_points = frechet_estimator.rand(24)
            euclidean_plot = EuclideanPlot(np_points)
            euclidean_plot.show()
            euclidean_mean = FrechetEstimator.euclidean_mean(np_points)
            logging.info(f'\nEuclidean mean: {euclidean_mean}')
            self.assertTrue(True)
        except InformationGeometricException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_estimate_SO3(self):
        try:
            manifold = SpecialOrthogonal(n=3, point_type="matrix")
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights=None)
            manifold_points = frechet_estimator.rand(4)

            so3_plot = SO3Plot(manifold_points)
            so3_plot.show()
            frechet_mean = frechet_estimator.estimate(manifold_points)
            euclidean_mean = FrechetEstimator.euclidean_mean(manifold_points)
            logging.info(f'\nFrechet mean:   {frechet_mean}\nEuclidean mean: {euclidean_mean}')

            self.assertTrue(True)
        except InformationGeometricException as e:
            logging.info(str(e))
            self.assertTrue(False)









