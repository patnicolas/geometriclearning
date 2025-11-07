__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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

from typing import AnyStr, Self
import networkx as nx
from torch_geometric.data import Data
__all__ = ['NetworkxGraph']

class NetworkxGraph(object):
    """
    Wrapper to initialize a graph using NetworkX
    This class has 2 constructors
        __init__:  Data => NetworkX graph
        build: Dataset name => NetworkX graph
    """
    def __init__(self, data: Data) -> None:
        """
        Constructor for the generation of graph from data
        @param data: Graph data
        @type data: torch_geometric.Data
        """
        super(NetworkxGraph, self).__init__()

        # Create a NetworkX graph
        G = nx.Graph()
        # Populate with the node from the dataset
        G.add_nodes_from(range(data.num_nodes))
        # Populate with the edges from the dataset: We need to transpose the tensor from 2 x num edges shape to
        # num edges x 2 shape
        edge_idx = data.edge_index.cpu().T
        G.add_edges_from(edge_idx.tolist())
        self.G = G

    @classmethod
    def build(cls, dataset_name: AnyStr) -> Self:
        """
        Alternative constructor for a NetworkX graph using the name of one of the
        PyTorch Geometric dataset
        @param dataset_name: Name of the dataset from PyTorch Geometric library
        @type dataset_name: str
        @return: Instance of NetworkX graph
        @rtype: NetworkxGraph
        """
        from dataset.graph.pyg_datasets import PyGDatasets

        # The class PyGDatasets validate the dataset is part of PyTorch Geometric Library
        pyg_dataset = PyGDatasets(dataset_name)
        dataset = pyg_dataset()
        return cls(dataset[0])

    def __str__(self) -> AnyStr:
        return str(self.G)
