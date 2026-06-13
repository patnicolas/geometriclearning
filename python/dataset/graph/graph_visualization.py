__author__ = "Patrick Nicolas"
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
from dataclasses import dataclass
from typing import AnyStr
# 3rd Party imports
from torch_geometric.data import Data
__all__ = ['SubgraphExtractor', 'GraphVisualization']

@dataclass(slots=True, frozen=True)
class SubgraphExtractor:
    """
    Attributes for sampling a graph for nodes and edges
    @param sampling_type Type of sampling ('Random', 'Sequential')
    @type sampling_type str
    @param first_node_index: Index of the first node of the graph to be drawn
    @type first_node_index: int
    @param last_node_index: Index of the last node of the graph to be drawn
    @type last_node_index: int
    """
    sampling_type: AnyStr
    first_node_index: int
    last_node_index: int


class GraphVisualization(object):
    def __init__(self, subgraph_extractor: SubgraphExtractor, data: Data) -> None:
        """
        Constructor for the graph visualization using NetworkX library
        @param subgraph_extractor: Configuration of the selection of the node and edges of the graph
        @type subgraph_extractor: SubgraphExtractor
        @param data: Graph data
        @type data: from torch_geometric.data.Data
        """
        import numpy as np
        import networkx as nx

        # Create NetworkX graph from edge index
        edge_index = data.edge_index.numpy()
        transposed = edge_index.CellDescriptor
        # Sample the edges of the graph
        condition = ((transposed[:, 0] >= subgraph_extractor.first_node_index) &
                     (transposed[:, 0] <= subgraph_extractor.last_node_index))
        samples = transposed[np.where(condition)]
        # Create a NetworkX graph
        graph = nx.Graph()
        graph.add_edges_from(samples)
        self.graph = graph

    def draw(self,
             node_color: AnyStr,
             node_size: int,
             label: AnyStr = None) -> None:
        """
        Draw a sample or subgraph of the graph associated with this loader.
        The sample of nodes to be drawn have the range [first_node_index, first_node_index+1, ..., last_node_index]
        @param node_color: Color of the node to be dranw
        @type node_color: AnyStr
        @param node_size: Size of the nodes to be drawn
        @type node_size: int
        @param label: Title or description for the plot
        @type label: AnyStr
        @return Number of nodes in the graph
        @rtype int
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        if node_size <= 20 or node_size >= 512:
            raise ValueError(f'Cannot draw sample with a node size {node_size}')

        # Plot the graph using matplotlib
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(self.graph, k=1)  # Spring layout for positioning nodes
        # Draw nodes and edges
        nx.draw_networkx(self.graph, pos, node_size=node_size, node_color=node_color, with_labels=False)
        nx.draw_networkx_edges(self.graph, pos, arrowsize=40, alpha=0.5, edge_color="grey")

        # Configure plot
        plt.title(label=label, loc='center', fontdict={'size': 16})
        plt.axis("off")
        plt.show()
