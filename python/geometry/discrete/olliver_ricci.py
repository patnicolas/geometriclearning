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
# 3rd Party imports
import torch
# Library imports
__all__ = ['OlliverRicci']


class OlliverRicci(object):
    fudge_factor = 1e-8
    default_early_stop_threshold = 5e-9

    def __init__(self,
                 alpha: float,
                 entropy_reg: float,
                 n_iters: int,
                 early_stop_threshold: float = default_early_stop_threshold) -> None:
        self.alpha = alpha
        self.entropy_reg = entropy_reg
        self.n_iters = n_iters
        self.early_stop_threshold = early_stop_threshold

    def sinkhorn_knopp(self,
                       r: torch.Tensor,
                       c: torch.Tensor,
                       cost_matrix: torch.Tensor,
                       early_stop_threshold: float) -> (int, torch.Tensor):
        """
        Computes the Sinkhorn approximation of the Wasserstein-1 distance.
        """
        # K is the kernel matrix  K = exp(-M/epsilon)
        K = torch.exp(-cost_matrix / self.entropy_reg)
        u = torch.ones_like(r) / r.shape[0]
        iters = self.n_iters

        # Normalization with a fudge factor to ensure convexity
        for i in range(self.n_iters):
            u_prev = u
            v = c / (torch.matmul(K.t(), u) + OlliverRicci.fudge_factor)
            u = r / (torch.matmul(K, v) + OlliverRicci.fudge_factor)
            if torch.abs(u - u_prev).max() < early_stop_threshold:
                iters = i
                break

        optimal_transport = torch.sum(u * torch.matmul(K * cost_matrix, v))
        return iters, optimal_transport

    def compute(self, adj_matrix: torch.Tensor, shortest_paths, ) -> torch.Tensor:
        """
        adj_matrix: (N, N) tensor
        shortest_paths: (N, N) tensor of geodesic distances
        alpha: mass to stay at the current node
        """
        n = adj_matrix.shape[0]
        curvature = torch.zeros_like(adj_matrix)

        # Define neighborhood distributions m_x
        # m_x(v) = alpha if v=x, else (1-alpha)/degree if v is neighbor
        degrees = adj_matrix.sum(dim=1)
        eye = torch.eye(n, device=adj_matrix.device)

        # Probability measures for all nodes: (N, N)
        m = (self.alpha * eye) + ((1 - self.alpha) * adj_matrix / degrees.unsqueeze(1))

        # Calculate curvature for each existing edge
        edges = torch.nonzero(adj_matrix)
        for u, v in edges:
            w1 = self.sinkhorn_knopp(m[u], m[v], shortest_paths)
            dist_uv = shortest_paths[u, v]
            curvature[u, v] = 1 - (w1 / dist_uv)
        return curvature
