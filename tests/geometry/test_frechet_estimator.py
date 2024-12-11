import unittest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import GradientDescent
from torch import Tensor

from geometry.frechet_estimator import FrechetEstimator
from geometry import GeometricException

from geometry.visualization.hypersphere_plot import HyperspherePlot
from geometry.visualization.euclidean_plot import EuclideanPlot
from geometry.visualization.so3_plot import SO3Plot


class FrechetEstimatorTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init_1(self):
        manifold = Hypersphere(dim=2, intrinsic=False, equip=True)
        weights = Tensor([0.4, 0.6, 1.0])
        X = Tensor([[0.4, -0.4, 0.6], [0.1, -1.0, 0.7], [0.7, -0.5, 0.0]])
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights)
            mean = frechet_estimator.estimate(X)
            print(mean)
            self.assertTrue(False)
        except GeometricException as e:
            print(str(e))
            self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_estimate_hypersphere(self):
        manifold = Hypersphere(dim=2)
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights=None)
            np_points = frechet_estimator.rand(8)

            hypersphere_plot = HyperspherePlot(np_points)
            # hypersphere_plot.show()
            frechet_mean = frechet_estimator.estimate(np_points)

            euclidean_plot = EuclideanPlot(np_points)
            euclidean_plot.show()
            euclidean_mean = FrechetEstimator.euclidean_mean(np_points)
            print(f'\nFrechet mean:   {frechet_mean}\nEuclidean mean: {euclidean_mean}')

            self.assertTrue(True)
        except GeometricException as e:
            print(str(e))
            self.assertTrue(False)


    def test_estimate_Euclidean(self):
        manifold = Hypersphere(dim=2)
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights=None)
            np_points = frechet_estimator.rand(24)
            euclidean_plot = EuclideanPlot(np_points)
            euclidean_plot.show()
            euclidean_mean = FrechetEstimator.euclidean_mean(np_points)
            print(f'\nEuclidean mean: {euclidean_mean}')
            self.assertTrue(True)
        except GeometricException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_estimate_SO3(self):
        manifold = SpecialOrthogonal(n=3, point_type="matrix")
        try:
            frechet_estimator = FrechetEstimator(manifold, GradientDescent(), weights=None)
            manifold_points = frechet_estimator.rand(4)

            so3_plot = SO3Plot(manifold_points)
            so3_plot.show()
            frechet_mean = frechet_estimator.estimate(manifold_points)
            euclidean_mean = FrechetEstimator.euclidean_mean(manifold_points)
            print(f'\nFrechet mean:   {frechet_mean}\nEuclidean mean: {euclidean_mean}')

            self.assertTrue(True)
        except GeometricException as e:
            print(str(e))
            self.assertTrue(False)









