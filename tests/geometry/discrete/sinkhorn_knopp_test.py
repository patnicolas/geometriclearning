import unittest
import logging
import python
import torch

from geometry.discrete import WassersteinException
from geometry.discrete.sinkhorn_knopp import SinkhornKnopp


class SinkhornKnoppTest(unittest.TestCase):

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

    def test_call(self):
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

            n_iters = 200
            early_stop_threshold = 0.01
            actual_n_iters, optimal_transport = sinkhorn_knopp(n_iters=n_iters, early_stop_threshold=early_stop_threshold)
            logging.info(f'\nConverged {actual_n_iters < n_iters}\nOptimal transport: {optimal_transport}')
            self.assertTrue(True)
        except (ValueError, WassersteinException) as e:
            logging.error(e)
            self.assertFalse(True)