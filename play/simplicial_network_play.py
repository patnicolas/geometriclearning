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

from typing import Dict, AnyStr, Any, Tuple, List

from omegaconf import DictConfig
from topobench.data.loaders.graph.tu_datasets import TUDatasetLoader
import numpy as np
from topomodelx.nn.simplicial.sccn import SCCN
from topomodelx.nn.simplicial.scn2 import SCN2
from topomodelx.utils.sparse import from_sparse
from toponetx import SimplicialComplex

from play import Play
import toponetx as tnx
import torch
import torch.nn as nn



class SimplicialConvolutionalNetwork(nn.Module):
    def __init__(self, channels: int, out_channels: int, num_layers: int) -> None:
        super(SimplicialConvolutionalNetwork, self).__init__()
        self.sccn = SCCN(channels=channels, max_rank=3, n_layers=num_layers, update_func="sigmoid")
        self.linear = nn.Linear(channels, out_channels)

    def forward(self, features: torch.Tensor, incidences: torch.Tensor, adjacencies: torch.Tensor) -> torch.Tensor:
        features = self.base_model(features, incidences, adjacencies)
        x = self.linear(features["rank_0"])
        return torch.softmax(x, dim=1)

    @staticmethod
    def create_incidences(complex: SimplicialComplex, max_rank: int) -> Dict[AnyStr, torch.Tensor]:
        return {
            f"rank_{rk+1}": from_sparse(complex.incidence_matrix(rank=rk+1))
            for rk in range(max_rank)
        }

    @staticmethod
    def create_adjacencies(complex: SimplicialComplex, max_rank: int) -> Dict[AnyStr, torch.Tensor]:
        adjacencies = {}
        adjacencies['rank_0'] = from_sparse(complex.incidence_matrix(rank=0)) + torch.eye(complex.shape[0]).to_sparse()

    @staticmethod
    def __sparse_identity(cplx: SimplicialComplex, rk: int) -> torch.Tensor:
        return torch.eye(cplx.shape[rk]).to_sparse()


if __name__ == '__main__':
    n_layers = 2
    n_out_channels = 2
    dataset: SimplicialComplex = tnx.datasets.karate_club(complex_type='simplicial', feat_dim=2)
