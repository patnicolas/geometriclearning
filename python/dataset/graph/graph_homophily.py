__author__ = "Patrick Nicolas"
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

# Standard Library imports
from typing import AnyStr, Self
from enum import StrEnum, verify, UNIQUE
# 3rd Party imports
from torch_geometric.data import Data
# Library imports
from dataset.graph.pyg_datasets import PyGDatasets
__all__ = ['GraphHomophily', 'GraphHomophilyType']


@verify(UNIQUE)
class GraphHomophilyType(StrEnum):
    """
    Enumerator for the 3 type of homophily ratio
    - Node homophily
    - Edge homophily
    - Class insensitive edge homophily
    """
    Node = 'node'
    Edge = 'edge'
    ClassInsensitiveEdge = 'edge_insensitive'


class GraphHomophily(object):
    """
        Implementation of the computation of the homophily of a graph using two methods
        - torch_geometric (__call__)
        - homegrown (compute) includes for reference...

        The main purpose of computing the homophily of a graph is to evaluate or predict the quality of prediction
        of a given graph neural network

        This class has two constructors:
        __init__ Default constructor with graph data and type of homophily computation
        build Alternative constructor for data set predefined in the Torch Geometric library

        Edge homophily
        The fraction of edges in a graph which connects nodes that have the same class label:
        math::
            \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }{|\mathcal{E}|}

        Node homophily:
        Edge homophily is normalized across neighborhoods
        math::
            \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w \in \mathcal{N}(v)
            \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }

        class insensitive edge homophily ratio
        Edge homophily is modified to be insensitive to the number of classes and size of each class
        math::
            \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|} {|\mathcal{V}|} \right)
    """

    def __init__(self, data: Data, homophily_type: GraphHomophilyType) -> None:
        """
        Default constructor for computing the homophily of a graph
        @param data: Graph data
        @type data: torch_geometric.data.Data
        @param homophily_type: Type of computation to perform (node, edge, edge insensitive
        @type homophily_type: Enum
        """
        self.data = data
        self.homophily_type = homophily_type

    @classmethod
    def build(cls, dataset_name: AnyStr, homophily_type: GraphHomophilyType) -> Self:
        """
        Alternative constructor that automatically loads a dataset using predefined in PyTorch Geometric and
        leverage the PyGDatasets class
        @param dataset_name: Name of geometric data set supported by PyGDatasets class
        @type dataset_name: AnyStr
        @param homophily_type: One of the 3 type of homophily (node, edge, and class insensitive edge)
        @type homophily_type: Enum
        @return: Instance of the GraphHomophily
        @rtype: GraphHomophily
        """
        pyg_dataset = PyGDatasets(dataset_name)
        dataset = pyg_dataset()
        return cls(dataset[0], homophily_type)

    def __call__(self) -> float:
        """
        Method that computes the homophily of a graph node or edge using the Torch Geometric API
        @return: Homophily ratio for the type of computation defined in the constructor
        @rtype: float
        """
        from torch_geometric.utils import homophily
        return homophily(edge_index=self.data.edge_index, y=self.data.y, method=self.homophily_type.value)

    def compute(self) -> float:
        """
        Method that computes the homophily of a graph node or edge using homegrown method defined as private method
        @return: Homophily ratio for the type of computation defined in the constructor
        @rtype: float
        """
        match self.homophily_type:
            case GraphHomophilyType.Node:
                return self.__node_homophily()
            case GraphHomophilyType.Edge:
                return self.__edge_homophily()
            case GraphHomophilyType.Edge:
                return self.__edge_homophily()

    def __str__(self) -> AnyStr:
        """
        Textual description of this class
        @return: Description homophily type and graph data
        @rtype: str
        """
        return f'\nHomophily type: {self.homophily_type.value}\n{self.data}'

    """ -----------------------    Private Helper Methods ---------------- """

    def __node_homophily(self) -> float:
        # Create adjacency list (dictionary of neighbors)
        adj = [[] for _ in range(self.data.num_nodes)]
        for src, dst in self.data.edge_index.t():
            adj[src.item()].append(dst.item())
            adj[dst.item()].append(src.item())

        homophily_per_node = []
        for node in range(self.data.num_nodes):
            neighbors = adj[node]
            # Discard isolated node
            if len(neighbors) > 0:
                same_label_count = sum(self.data.y[node] == self.data.y[neighbor]
                                       for neighbor in neighbors)
                homophily_per_node.append(same_label_count / len(neighbors))
        # handle case where all nodes are isolated
        return 0.0 if len(homophily_per_node) == 0 \
            else sum(homophily_per_node) / len(homophily_per_node)

    def __edge_homophily(self) -> float:
        # Get the vertices id for each of the pair of edge_index
        matches = sum([1 for edge_idx in self.data.edge_index
                       if self.data.y[edge_idx[0]] == self.data.y[edge_idx[1]]])
        # Compute the ratio
        return matches / self.data.num_nodes

