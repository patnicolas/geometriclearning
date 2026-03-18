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
from typing import Dict, AnyStr, Any, Literal
# 3rd Party library imports
import torch
import torch.nn as nn
# Library imports
from topology.topo_bench_wrapper import TopoBenchWrapper
from play import Play


class HypergraphModel(nn.Module):
    """
    Simple two layer perceptron to process the graph nodes and edges for evaluation with TUDataset graph data
    """
    def __init__(self, in_channels: int, dimension_hidden: int, out_channels: int) -> None:
        super().__init__()
        # Linear + Activation modules for the Hypergraph nodes
        self.hypernodes_linear = nn.Linear(in_channels, dimension_hidden)
        self.hypernodes_relu = nn.ReLU()

        # Linear + Activation modules for Hyperedges
        self.hyperedges_linear = nn.Linear(in_channels, dimension_hidden)
        self.hyperedges_relu = nn.ReLU()
        self.out_channels = out_channels

    def get_in_channels(self) -> int:
        return self.hypernodes_linear.in_features

    def get_dim_hidden(self) -> int:
        return self.hypernodes_linear.out_features

    def forward(self, batch) -> Dict[AnyStr, Any]:
        # Step 1 Extract hypernodes and hyperedges from the current batch
        x_hypernodes = batch.x
        x_hyperedges = torch.sparse.mm(batch.incidence_hyperedges, x_hypernodes)

        # Step 2 Process features from hypernodes
        x_hypernodes = self.hypernodes_relu(self.hypernodes_linear(x_hypernodes))

        # Step 3 Process features from hyperedges
        x_hyperedges = self.hyperedges_relu(self.hyperedges_linear(x_hyperedges))
        return {"labels": batch.y, "batch_0": batch.batch_0, "x_0": x_hypernodes, "hyperedge": x_hyperedges}


class TopoBenchPlay(Play):
    """
     This class evaluates the basic functionality of TopoBench framework through a parameterized and componentized
     wrapper, TopoBenchWrapper.

     TopoBench is a modular Python framework built to standardize benchmarks and streamline research within
     Topological Deep Learning (TDL). It enables the seamless training and comparative analysis of various Topological
     Neural Networks (TNNs) across multiple domains, including graphs, simplicial complexes, cellular complexes, and
     hypergraphs.

    The main method, play,  implements the evaluation code of the substack article "Benchmarking Topological Deep
    Learning".

    The features are implemented by the classes TopoBenchWrapper and TopoBenchConfig in the source file
                    python/topology/Topo_bench_config.py and python/topology/Topo_bench_wrapper.py
    The class TopoBenchPlay is a wrapper of the class TopoBenchWrapper
    """
    def __init__(self, lr: float, float_precision: Literal[16, 32, 64]) -> None:
        super(TopoBenchPlay, self).__init__()
        self.lr = lr
        self.float_precision = float_precision

    def play(self) -> None:
        dataset_name = 'MUTAG'
        hypergraph_model = HypergraphModel(dimension_hidden=16, in_channels=7, out_channels=2)
        topo_bench_wrapper = TopoBenchWrapper.build(graph_network=hypergraph_model,
                                                    k_value=1,
                                                    data_name=dataset_name,
                                                    lr=self.lr)
        topo_bench_wrapper.train(max_epochs=48, float_precision=self.float_precision)

        dataset_name = 'PROTEINS'
        hypergraph_model = HypergraphModel(dimension_hidden=24, in_channels=3, out_channels=2)
        topo_bench_wrapper = TopoBenchWrapper.build(graph_network=hypergraph_model,
                                                    k_value=1,
                                                    data_name=dataset_name,
                                                    lr=self.lr)
        topo_bench_wrapper.train(max_epochs=48, float_precision=32)


if __name__ == '__main__':
    topo_bench_play = TopoBenchPlay(lr=0.0005, float_precision=32)
    topo_bench_play.play()









