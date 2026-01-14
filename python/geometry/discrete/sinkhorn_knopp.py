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
from typing import AnyStr, Self, Tuple
# 3rd Party imports
import torch
# Library imports
from geometry.discrete import Wasserstein1Approximation, WassersteinException
from geometry.discrete.floyd_warshall import FloydWarshall

__all__ = ['SinkhornKnopp']


class SinkhornKnopp(Wasserstein1Approximation):
    """
    The Sinkhorn-Knopp algorithm provides an efficient way to compute the Sinkhorn distance by solving an
    entropic-regularized optimal transport problem.

    Given an optimal transport plan P, the cost matrix M,  the entropy regulation factor epsilon, a source distribution
    r and a destination distribution c.
    .. math::
        \begin{matrix}
        K=e^{-\frac{M}{\epsilon}}  \\
        u=\frac{r}{Kv}  \\
        v=\frac{r}{K^{T}v} \\
        P=diag(u).K.diag(v)
        \end{matrix}
    """
    fudge_factor = 1e-9

    def __init__(self, r: torch.Tensor, c: torch.Tensor, cost_matrix: torch.Tensor, epsilon: float) -> None:
        """
        Default constructor for a user-provided cost matrix. The constructor throws two types of exceptions:
        - ValueError if constructor arguments are out of bounds
        - WassersteinException if the entropy regularization is inadequate for computing the Earth Mover's distance
                epsilon < mean(cost_matrix)/10

        @param r: Source probability distribution
        @type r:  Torch Tensor
        @param c: Destination probability distribution
        @type c:  Torch Tensor
        @param cost_matrix: Cost matrix computed as all-pairs shortest paths in the graph
        @type cost_matrix:
        @param epsilon:  Entropy regularization
        @type epsilon: float
        """
        if r.shape[0] != cost_matrix.shape[0]:
            raise ValueError(f'Shape of r distribution {r.shape[0]} should match cost_matrix {cost_matrix.shape[0]}')
        if c.shape[0] != cost_matrix.shape[1]:
            raise ValueError(f'Shape of c distribution {c.shape[0]} should match cost_matrix {cost_matrix.shape[1]}')

        r, c = SinkhornKnopp.__normalize_r_c(r, c)
        super(SinkhornKnopp, self).__init__(r, c)

        if epsilon < 0.001 or epsilon > 0.5:
            raise ValueError(f'Entropy regularization {epsilon} should be [0.001, 0.5]')
        if epsilon >= torch.sum(cost_matrix):
            raise WassersteinException(f"Entropy regularization {epsilon} inadequate for Earth Mover's distance")

        self.cost_matrix = SinkhornKnopp.__normalize_cost_matrix(cost_matrix)
        self.epsilon = epsilon

    @classmethod
    def build(cls, r: torch.Tensor, c: torch.Tensor, floyd_warshall: FloydWarshall, epsilon: float) -> Self:
        """
        Alternative constructor for approximating the one dimensional Wasserstein distance. It takes a source
        probability distribution r, a destination probability distribution c, the Floyd-Warshall all-pairs
        shortest paths and entropy regularization factor.

        @param r: Marginal probability distribution for source vertext
        @type r: Torch Tensor
        @param c: Marginal probability distribution for destination vertex
        @type c: Torch Tensor
        @param floyd_warshall: Floyd-Warshall algorithm to compute the all-pairs shortest distances
        @type floyd_warshall: FloydWarshall
        @param epsilon: Entropy regularization
        @type epsilon: float
        @return: Instance of this class
        @rtype: SinkhornKnopp
        """
        cost_matrix = floyd_warshall()
        return cls(r, c, cost_matrix, epsilon)

    def __str__(self) -> AnyStr:
        return (f'\nSource distribution: {self.r}\nTarget distribution {self.c}\nCost matrix {self.cost_matrix}'
                f'\nEntropy regularization factor: {self.epsilon}')

    def __call__(self, n_iters: int, early_stop_threshold: float) -> (int, torch.Tensor):
        """
        Execute the iteration on the formula described in the class doc string.
        Pseudo-code:
            compute K <- epsilon, M
            init v <- [1]
            DO
                u <- r/K.v
                v <- c/KT.u
            WHILE u < threshold and v < threshold
        P = diag(u).K.diag(v)

        @param n_iters: Maximum number of iterations
        @type n_iters: int
        @param early_stop_threshold: Convergence condition
        @type early_stop_threshold: float
        @return: Tuple (Actual number of iterations, Approximate Wasserstein distance)
        @rtype: Tuple[int, Tensor]
        """
        if early_stop_threshold < 1e-8 or early_stop_threshold > 0.5:
            raise ValueError(f'Early stop threshold {early_stop_threshold} should be [1e-8, 0.5]')
        
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

        # Clean from init values
        self.cost_matrix[self.cost_matrix > 1e+5] = 0.0
        optimal_transport = torch.sum(u * torch.matmul(K * self.cost_matrix, v))
        return iters, optimal_transport

    """ -----------------  Private Helper Method ----------------------------- """

    @staticmethod
    def __normalize_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(cost_matrix)
        min_value = torch.min(cost_matrix)
        return (cost_matrix - min_value)/(max_value - min_value) \
            if max_value.item() != 1.0 and min_value.item() != 0.0 else cost_matrix

    @staticmethod
    def __normalize_r_c(r: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if sum(r) < 0.999 or sum(r) > 1.001:
            r = r / sum(r)
        if sum(c) < 0.999 or sum(c) > 1.001:
            c = c / sum(c)
        return r, c
