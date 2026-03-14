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

# Python Standard Library imports
from typing import Dict, AnyStr, Any,  Literal
import logging
# 3rd Party library imports
import numpy as np
import torch
import torch.nn as nn
# Library imports
from metric.metric_type import MetricType
from topology.topo_bench_wrapper import TopoBenchWrapper
from plots.metric_plotter import MetricPlotterParameters


class HypergraphModel(nn.Module):
    def __init__(self, in_channels: int, dimension_hidden: int, out_channels: int) -> None:
        super().__init__()
        # Linear module for the Hypergraph nodes
        self.linear_hypernodes = torch.nn.Linear(in_channels, dimension_hidden)
        # Linear module for Hyperedges
        self.linear_hyperedges = torch.nn.Linear(in_channels, dimension_hidden)
        self.out_channels = out_channels

    def get_in_channels(self) -> int:
        return self.linear_hypernodes.in_features

    def get_dim_hidden(self) -> int:
        return self.linear_hypernodes.out_features

    def forward(self, batch) -> Dict[AnyStr, Any]:
        # Step 1 Extract hypernodes and hyperedges from the current batch
        x_hypernodes = batch.x
        x_hyperedges = torch.sparse.mm(batch.incidence_hyperedges, x_hypernodes)
        # Step 2 Process features from hypernodes
        x_hypernodes = self.linear_hypernodes(x_hypernodes)
        x_hypernodes = torch.relu(x_hypernodes)
        # Step 3 Process features from hyperedges
        x_hyperedges = self.linear_hyperedges(x_hyperedges)
        x_hyperedges = torch.relu(x_hyperedges)
        return {"labels": batch.y, "batch_0": batch.batch_0, "x_0": x_hypernodes, "hyperedge": x_hyperedges}


if __name__ == '__main__':
    model = HypergraphModel(dimension_hidden=16, in_channels=7, out_channels=2)
    topo_bench_wrapper = TopoBenchWrapper.build(graph_network=model, k_value=1, data_name='MUTAG', lr=0.0005)
    topo_bench_wrapper.train(max_epochs=48, float_precision=32)

    model = HypergraphModel(dimension_hidden=24, in_channels=3, out_channels=2)
    topo_bench_wrapper = TopoBenchWrapper.build(graph_network=model, k_value=2, data_name='PROTEINS', lr=0.0005)
    topo_bench_wrapper.train(max_epochs=64, float_precision=32)

    """ ------------------------------------------------------------------------------- """









