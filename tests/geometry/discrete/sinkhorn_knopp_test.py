import unittest
import logging
import python
import torch

from geometry.discrete import WassersteinException
from geometry.discrete.sinkhorn_knopp import SinkhornKnopp


class SinkhornKnoppTest(unittest.TestCase):

    # @unittest.skip('Ignored')
    def test_init(self):
        try:
            # 1. Source distribution (r) - Must sum to 1
            # Represents probabilities across 3 source points
            r = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)

            # 2. Destination distribution (c) - Must sum to 1
            # Represents probabilities across 4 target points
            c = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float32)

            # 3. Cost Matrix (M) - Size (n_source, n_target)
            # Represents the "distance" or cost to move mass from r[i] to c[j]
            M = torch.tensor([
                [2.0, 4.2, 1.5, 3.0],
                [1.0, 0.5, 2.2, 4.0],
                [5.0, 3.0, 0.8, 1.2]
            ], dtype=torch.float32)
            sinkhorn_knopp = SinkhornKnopp(r=r, c=c, cost_matrix=M, epsilon=0.05)
            logging.info(sinkhorn_knopp)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    # @unittest.skip('Ignored')
    def test_call(self):
        try:
            # 1. Source distribution (r) - Must sum to 1
            # Represents probabilities across 3 source points
            r = torch.tensor([0.2, 0.4, 0.3, 0.1], dtype=torch.float32)

            # 2. Destination distribution (c) - Must sum to 1
            # Represents probabilities across 4 target points
            c = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float32)

            # 3. Cost Matrix (M) - Size (n_source, n_target)
            # Represents the "distance" or cost to move mass from r[i] to c[j]
            M = torch.tensor([
                [2.0, 4.2, 1.5, 3.0],
                [1.0, 0.5, 2.2, 4.0],
                [5.0, 3.0, 0.8, 1.2],
                [0.0, 1.1, 0.0, 2.1]
            ], dtype=torch.float32)
            sinkhorn_knopp = SinkhornKnopp(r=r, c=c, cost_matrix=M, epsilon=0.05)

            n_iters = 200
            early_stop_threshold = 0.01
            actual_n_iters, optimal_transport = sinkhorn_knopp(n_iters=n_iters, early_stop_threshold=early_stop_threshold)
            logging.info(f'\nConverged {actual_n_iters < n_iters}\nOptimal transport: {optimal_transport}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignored')
    def test_earth_movers_distance(self):
        from torch.distributions import LogNormal, Normal

        x = torch.linspace(-30.0, 44.0, 50)

        normal = Normal(torch.Tensor([8.6]), torch.Tensor([2.3]))
        normal_samples = normal.log_prob(x).exp()
        logging.info(normal_samples)

        log_normal = Normal(torch.Tensor([16.0]), torch.Tensor([7.3]))
        normal = Normal(torch.Tensor([2.0]), torch.Tensor([9.3]))
        normal_samples = normal.log_prob(x).exp()
        log_normal_samples = 0.1*(9*log_normal.log_prob(x).exp() + normal_samples)
        logging.info(f'LogNormal: {log_normal_samples}')

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 8))
        fig.set_facecolor('#F2F9FE')

        plt.plot(x, normal_samples)
        plt.fill_between(x, normal_samples, alpha=0.5, color='red')
        plt.plot(x, log_normal_samples)
        plt.fill_between(x, log_normal_samples, alpha=0.5, color='blue')
        plt.title("Earth Mover's Distance", fontdict={'family':'serif', 'size': 24})
        plt.tick_params(axis='both', labelsize=14)
        plt.annotate('Earth Moving', xy=(-6, 0.048), fontsize=18)
        plt.show()
