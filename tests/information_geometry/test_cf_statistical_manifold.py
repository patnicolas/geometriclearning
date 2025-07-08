import unittest
import logging
import torch
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import python
from python import SKIP_REASON

from geomstats.information_geometry.binomial import BinomialDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.poisson import PoissonDistributions
from information_geometry.cf_statistical_manifold import CFStatisticalManifold
from typing import Tuple, AnyStr
import logging
import python



class CFStatisticalManifoldTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        try:
            exponential_distributions = ExponentialDistributions(equip=True)
            statistical_manifold = CFStatisticalManifold(exponential_distributions, (1.0, 2.0))
            logging.info(f'\n{statistical_manifold=}')
            self.assertTrue(statistical_manifold.fisher_rao_metric.signature == (1, 0))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_random_samples(self):
        try:
            exponential_distributions = ExponentialDistributions(equip=True)
            exponential_manifold = CFStatisticalManifold(exponential_distributions, (-2.0, 2.0))
            exponential_samples = exponential_manifold.samples(n_samples=8)
            assert exponential_manifold.belongs(exponential_samples)
            geometric_distributions = GeometricDistributions(equip=True)
            geometric_manifold = CFStatisticalManifold(geometric_distributions, (1, 10))
            geometric_samples = geometric_manifold.samples(n_samples=4)
            exponential_samples_str = '\n'.join([str(x) for x in exponential_samples])
            exponential_geometric_str = '\n'.join([str(x) for x in geometric_samples])
            logging.info(f'\nExponential Distribution Manifold 8 random samples \n{exponential_samples_str}'
                         f'\nGeometric Distribution Manifold 4 random samples\n{exponential_geometric_str}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_some_fisher_metrics(self):
        try:
            exponential_manifold = CFStatisticalManifold(ExponentialDistributions(equip=True), (0.0, 2.0))
            poisson_manifold = CFStatisticalManifold(PoissonDistributions(equip=True), (0, 20))
            exponential_metric = exponential_manifold.metric_matrix()
            poisson_metric = poisson_manifold.metric_matrix()
            logging.info(f'\nExponential Distribution Fisher metric: {exponential_metric}'
                         f'\nPoisson Distribution Fisher metric: {poisson_metric}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_metric_matrix_exponential(self):
        try:
            exponential_distributions = ExponentialDistributions(equip=True)
            statistical_manifold = CFStatisticalManifold(exponential_distributions, (1.0, 2.0))
            metric = statistical_manifold.metric_matrix()
            logging.info(f'Fisher-Rao metric for exponential: {metric}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_exp_maps(self):
        try:
            exponential_manifold = CFStatisticalManifold(ExponentialDistributions(equip=False), (0.0, 2.0))
            tgt_vector = torch.Tensor([0.5])
            base_point = exponential_manifold.samples(1)
            exponential_end_point = exponential_manifold.exp(tgt_vector, base_point)

            geometric_manifold = CFStatisticalManifold(GeometricDistributions(equip=False), (1.0, 10.0))
            tgt_vector = torch.Tensor([0.5])
            base_point = geometric_manifold.samples(1)
            geometric_end_point = geometric_manifold.exp(tgt_vector, base_point)
            logging.info(f'\nExponential Distribution Manifold End point: {exponential_end_point}'
                         f'\nGeometric Distribution Manifold End point: {geometric_end_point}\n')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_log_maps(self):
        try:
            exponential_manifold = CFStatisticalManifold(ExponentialDistributions(equip=False), (0.0, 2.0))
            random_points = exponential_manifold.samples(2)
            base_point = random_points[0]
            manifold_point = random_points[1]
            exponential_vector = exponential_manifold.log(manifold_point, base_point)
            logging.info(f'\nExponential Distribution Manifold Tangent Vector\nBase:{base_point} to:{manifold_point}: '
                         f'{exponential_vector}')

            geometric_manifold = CFStatisticalManifold(GeometricDistributions(equip=False), (1.0, 10.0))
            random_points = geometric_manifold.samples(2)
            base_point = random_points[0]
            manifold_point = random_points[1]
            geometric_vector = geometric_manifold.log(manifold_point, base_point)
            logging.info(f'\nGeometric Distribution Manifold Tangent Vector\nBase:{base_point} to:{manifold_point}: '
                         f'{geometric_vector}\n')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_exp_map_geometric(self):
        try:
            geometric_distributions = GeometricDistributions(equip=True)
            statistical_manifold = CFStatisticalManifold(geometric_distributions, (1, 10))
            tgt_vector = torch.Tensor([0.5])
            base_point = statistical_manifold.samples(1)
            end_point = statistical_manifold.exp(tgt_vector, base_point)
            logging.info(end_point)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_exponentials(self):
        CFStatisticalManifoldTest.__visualize(distribution=ExponentialDistributions(equip=True),
                                              bounds=(0.0, 2.0),
                                              num_samples=128,
                                              param_desc="$ \\theta \in$")
        self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_poissons(self):
        CFStatisticalManifoldTest.__visualize(distribution=PoissonDistributions(equip=True),
                                              bounds=(0.0, 20.0),
                                              num_samples=128,
                                              param_desc='$\\lambda \in$')
        self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_binomial(self):
        n_draws = 40
        CFStatisticalManifoldTest.__visualize(distribution=BinomialDistributions(equip=True, n_draws=n_draws),
                                              bounds=(0.0, n_draws/2),
                                              num_samples=128,
                                              param_desc='n_draws=40,  p $ \in$')
        self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_geometric(self):
        CFStatisticalManifoldTest.__visualize(distribution=GeometricDistributions(equip=True),
                                              bounds=(1.0, 10.0),
                                              num_samples=128,
                                              param_desc='p $ \in$')
        self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_animate_geometric(self):
        CFStatisticalManifoldTest.__animate(distribution=GeometricDistributions(equip=True),
                                            bounds=(1.0, 10.0),
                                            num_samples=96,
                                            param_desc='p $ \in$')
        self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_normal_fixed_stddev(self):
        std = 1.0
        x = torch.normal(mean=0.0, std=std, size=(2000,))

        # Score function
        scores = (x - 0.0) / 1.0 ** 2
        # Compute mean squared score
        fisher_info = torch.mean(scores ** 2)
        metric = fisher_info.item()
        logging.info(f'Metric={metric}')
        self.assertTrue(-0.02 < metric - 1.0/std**2 < 0.02)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize_normal(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import norm
        from matplotlib.animation import FuncAnimation

        # Fixed value of x at which we evaluate the PDF
        x_val = 0.0
        num = 100

        # Grid of mean (mu) and standard deviation (sigma)
        mu_vals = np.linspace(-10, 10, num)
        sigma_vals = np.linspace(0.0, 1.5, num)
        MU, SIGMA = np.meshgrid(mu_vals, sigma_vals)

        # Evaluate normal PDF at x_val for each (mu, sigma) pair
        Z = norm.pdf(x_val, loc=MU, scale=SIGMA)

        # 3D Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame: int) -> None:
            if frame > 1 and frame % 3 == 0:
                mu = MU[0:frame]
                sigma = SIGMA[0:frame]
                surf = ax.plot_surface(mu, sigma, Z[0:frame], cmap='viridis', edgecolor='none')

        ax.set_title('PDF of Normal Distribution at origin')
        ax.set_xlabel('Mean (μ)')
        ax.set_ylabel('Standard Deviation (σ)')
        ax.set_zlabel('PDF value')
        ani = FuncAnimation(fig, update, frames=num, interval=8, repeat=False, blit=False)
        # plt.show()
        ani.save('normal_manifold_animation.mp4', writer='ffmpeg', fps=32, dpi=240)

    """ ---------------------------   Support methods ---------------------------  """

    @staticmethod
    def __visualize(distribution: InformationManifoldMixin,
                    bounds: Tuple[float, float],
                    num_samples: int,
                    param_desc: AnyStr) -> None:
        import torch

        statistical_manifold = CFStatisticalManifold(distribution, bounds=bounds)
        values = statistical_manifold.samples(num_samples)
        min_theta = f'{float(torch.min(values)):.4f}'
        max_theta = f'{float(torch.max(values)):.4f}'
        statistical_manifold.visualize_pdfs(
            values,
            rf"{distribution.__class__.__name__} {num_samples} samples {param_desc} [{min_theta}, {max_theta}]"
        )

    @staticmethod
    def __animate(distribution: InformationManifoldMixin,
                  bounds: Tuple[float, float],
                  num_samples: int,
                  param_desc: AnyStr) -> None:
        statistical_manifold = CFStatisticalManifold(distribution, bounds=bounds)
        values = statistical_manifold.samples(num_samples)
        min_theta = f'{float(torch.min(values)):.4f}'
        max_theta = f'{float(torch.max(values)):.4f}'
        statistical_manifold.animate(values, rf"{param_desc} [{min_theta}, {max_theta}]")

