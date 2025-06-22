_author__ = "Patrick Nicolas"
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

import networkx as nx
from networkx import Graph
from torch_geometric.data import Data
from typing import Tuple, AnyStr, Callable, Dict, Any, Self, List
import matplotlib.pyplot as plt


class GNNPlotter(object):
    def __init__(self, graph: Graph, data: Data, sampled_node_index_range: Tuple[int, int] = None) -> None:
        """
        Constructor for the visual representation of the data associated with a Graph Neural Network
        The graph is sampled is the parameter is defined. The nodes of the graph are arbitrary selected
        as follows:
        node indices [0, 1, ... sampled_node_index_range[0], ...,sampled_node_index_range[1], ...]

        @param graph: PyG directed or undirected graph instance
        @type graph: networkx.Graph
        @param data: Data from data set associated with graph neural network
        @type data: torch_geometric.data.Data
        @param sampled_node_index_range: Low and high bound of the indices of sampled nodes
        @type sampled_node_index_range: Tuple
        """
        assert GNNPlotter.__validate(sampled_node_index_range), \
            f'Incorrect indices for sampling graph nodes'
        self.graph = graph
        self.data = data
        self.sampled_node_index_range = sampled_node_index_range

    @classmethod
    def build(cls, data: Data, sampled_node_index_range: Tuple[int, int] = None) -> Self:
        """
        Constructor for the visual representation of the data associated with an undirected
        Graph Neural Network
        @param data: Data from data set associated with graph neural network
        @type data: torch_geometric.data.Data
        @param sampled_node_index_range: Low and high bound of the indices of sampled nodes
        @type sampled_node_index_range: Tuple
        @return: Instance of GNN plotter
        @rtype: GNNPlotter
        """
        return cls(nx.Graph(), data, sampled_node_index_range)

    @classmethod
    def build_directed(cls, data: Data, sampled_node_index_range: Tuple[int, int] = None) -> Self:
        """
        Constructor for the visual representation of the data associated with a directed
        Graph Neural Network
        @param data: Data from data set associated with graph neural network
        @type data: torch_geometric.data.Data
        @param sampled_node_index_range: Low and high bound of the indices of sampled nodes
        @type sampled_node_index_range: Tuple
        @return: Instance of GNN plotter
        @rtype: GNNPlotter
        """
        return cls(nx.DiGraph(), data, sampled_node_index_range)

    def len(self) -> int:
        return len(self.data.edge_index)

    def sample(self) -> int:
        """
        Sample/extract a subgraph from a large graph by selecting its nodes through a range of indices
        """
        import numpy as np

        # Create NetworkX graph from edge index
        edge_index = self.data.edge_index.numpy()
        transposed = edge_index.T
        # Sample the edges of the graph
        if self.sampled_node_index_range is not None:
            last_node_index = len(self.data.y) if self.sampled_node_index_range[1] >= len(self.data.y) \
                else self.sampled_node_index_range[1]
            condition = ((transposed[:, 0] >= self.sampled_node_index_range[0]) & (transposed[:, 0] <= last_node_index))
            sampled_nodes = transposed[np.where(condition)]
        else:
            sampled_nodes = transposed
        # Assign the samples to the edge of the graph
        self.graph.add_edges_from(sampled_nodes)
        return len(sampled_nodes)

    def draw(self,
             layout_func: Callable[[Graph], Dict[Any, Any]],
             node_color: AnyStr,
             node_size: int,
             title: AnyStr) -> int:
        """
        Draw a sample or subgraph of the graph associated with this loader.
        The sample of nodes to be drawn have the range [first_node_index, first_node_index+1, ..., last_node_index]
        @param layout_func: Layout of the graph as defined as a function Graph -> Dict[key, position]
        @type layout_func: Callable[[Graph], Dict[Any, Any]]
        @param node_color: Color of the node to be drawn
        @type node_color: AnyStr
        @param node_size: Size of the nodes to be drawn
        @type node_size: int
        @param title: Title of the plot
        @type title: str
        """
        num_sampled_nodes = self.sample()

        # Plot the graph using matplotlib
        plt.figure(figsize=(8, 8))
        # pos = nx.spring_layout(self.graph, k=1)  # Spring layout for positioning nodes
        # Draw nodes and edges
        pos = layout_func(self.graph)
        nx.draw(self.graph, pos, node_size=node_size, node_color=node_color)
        nx.draw_networkx_edges(self.graph, pos, arrowsize=40, alpha=0.5, edge_color="black")

        # Configure plot
        plt.title(title)
        plt.axis("off")
        plt.show()
        return num_sampled_nodes

    def draw_all(self,
                 node_size: int,
                 title: AnyStr) -> int:
        """
        Draw a sample or subgraph of the graph using a set of predefined layout.
        @param node_size: Size of the nodes to be drawn
        @type node_size: int
        @param title: Title of the plot
        @type title: str
        """
        layout_funcs = {
            'Spring layout': lambda graph: nx.spring_layout(graph, k=1),
            'Random layout': lambda graph: nx.random_layout(graph, center=None, dim=2),
            'Kamada-Kawai layout': lambda graph: nx.kamada_kawai_layout(graph),
            'Planar layout': lambda graph: nx.kamada_kawai_layout(graph),
            'Spiral layout': lambda graph: nx.spiral_layout(graph),
            'Arf layout': lambda graph: nx.arf_layout(graph)
        }

        num_sampled_nodes = self.sample()
        plt.figure(figsize=(10, 10))
        node_colors = ['blue', 'green', 'red', 'magenta', 'gray', 'orange']

        for idx, (plot_type, layout_func) in enumerate(layout_funcs.items(), 1):
            plt.subplot(3, 2, idx)
            pos = layout_func(self.graph)
            nx.draw(self.graph, pos, node_size=node_size, node_color=node_colors[idx-1])
            nx.draw_networkx_edges(self.graph, pos, arrowsize=18, alpha=0.5, edge_color="black")
            plt.title(f'{title}: {plot_type}', fontdict={'family': 'sans-serif', 'size': 15})
            plt.axis("off")
        plt.show()
        return num_sampled_nodes

    """ ---------------------  Private help methods ------------------------  """

    @staticmethod
    def __validate(sampled_node_index_range: Tuple[int, int]) -> bool:
        return sampled_node_index_range is None or 0 <= sampled_node_index_range[0] <= sampled_node_index_range[1]






