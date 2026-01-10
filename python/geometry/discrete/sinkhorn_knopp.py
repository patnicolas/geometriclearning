__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import AnyStr
# 3rd Party imports
import torch
# Library imports
from geometry.discrete import Wasserstein1Approximation
from geometry.discrete import WassersteinException

__all__ = ['SinkhornKnopp']


class SinkhornKnopp(Wasserstein1Approximation):
    fudge_factor = 1e-9

    def __init__(self, r: torch.Tensor, c: torch.Tensor, cost_matrix: torch.Tensor, epsilon: float) -> None:
        super(SinkhornKnopp, self).__init__(r, c)
        if sum(r) < 0.999 or sum(r) > 1.001:
            raise ValueError(f'Sum of the source distribution points {sum(r)} should be 1.0')
        if sum(c) < 0.999 or sum(c) > 1.001:
            raise ValueError(f'Sum of the target distribution points {sum(c)} should be 1.0')
        if epsilon < 0.001 or epsilon > 0.5:
            raise ValueError(f'Entropy regularization {epsilon} should be [0.001, 0.5]')
        if epsilon < 0.1*torch.mean(cost_matrix):
            raise WassersteinException(f"Entropy regularization {epsilon} inadequate for Earth Mover's distance")

        self.cost_matrix = SinkhornKnopp.__normalize_cost_matrix(cost_matrix)
        self.epsilon = epsilon

    def __str__(self) -> AnyStr:
        return (f'\nSource distribution: {self.r}\nTarget distribution {self.c}\nCost matrix {self.cost_matrix}'
                f'\nEntropy regularization factor: {self.epsilon}')

    def __call__(self, n_iters: int, early_stop_threshold: float) -> (int, torch.Tensor):
        """
        Computes the Sinkhorn approximation of the Wasserstein-1 distance.
        """
        # K is the kernel matrix  K = exp(-M/epsilon)
        K = torch.exp(-self.cost_matrix / self.epsilon)
        u = torch.ones_like(self.r) / self.r.shape[0]
        iters = n_iters

        # Normalization with a fudge factor to ensure convexity
        for i in range(n_iters):
            u_prev = u
            v = self.c / (torch.matmul(K.t(), u) + SinkhornKnopp.fudge_factor)
            u = self.r / (torch.matmul(K, v) + SinkhornKnopp.fudge_factor)
            if torch.abs(u - u_prev).max() < early_stop_threshold:
                iters = i
                break

        optimal_transport = torch.sum(u * torch.matmul(K * self.cost_matrix, v))
        return iters, optimal_transport

    """ -----------------  Private Helper Method ----------------------------- """

    @staticmethod
    def __normalize_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(cost_matrix)
        min_value = torch.min(cost_matrix)
        return (cost_matrix - min_value)/(max_value - min_value) \
            if max_value.item() != 1.0 and min_value.item() != 0.0 else cost_matrix
