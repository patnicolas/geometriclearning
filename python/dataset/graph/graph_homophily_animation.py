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

from typing import List
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib.animation as animation
from networkx import Graph
import networkx as nx
from networkx.classes.reportviews import NodeView
from util.base_animation import BaseAnimation
__all__ = ['GraphHomophilyAnimation']


class GraphHomophilyAnimation(BaseAnimation):
    def __init__(self,
                 logo_pos: List[float],
                 interval: int,
                 fps: int,
                 num_nodes: int,
                 average_degree: float, ) -> None:
        """

        @param logo_pos: Position of the chart used in call to plt.set_position or ax.set_position
        @type logo_pos: 4-dimension array
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        @param num_nodes:
        @type num_nodes:
        @param average_degree:
        @type average_degree:
        """
        super(GraphHomophilyAnimation, self).__init__(logo_pos, interval, fps)

        self.G = nx.erdos_renyi_graph(n=num_nodes, p=average_degree, seed=42)
        self.central_node: int = GraphHomophilyAnimation.__get_central_node(self.G)

    def draw(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('#f0f9ff')
        ax.set_facecolor('#f0f9ff')
        ax.set_position(self.chart_pos)
        self._draw_logo(fig)

        hop1_neighbors, hop2_neighbors = self.__build_neighbors()
        node_artists = self.__layout(ax)
        highlighted_nodes = [self.central_node]
        highlighted_nodes_2 = [self.central_node]

        def update(frame):
            new_colors = []
            if frame < len(hop1_neighbors):
                highlighted_nodes.append(hop1_neighbors[frame])
                ax.text(x=0.42,
                        y=0.76,
                        s=f"1st Hop: {len(highlighted_nodes)} neighbors",
                        fontdict={'fontsize': 14, 'fontname': 'Helvetica', 'fontweight': 'bold', 'color': 'orange'},
                        bbox=dict(facecolor='black', edgecolor='black'))
                new_colors = ['orange' if n in highlighted_nodes else 'lightgray' for n in self.G.nodes()]
                new_colors[self.central_node] = 'red'
            else:
                if frame - len(hop1_neighbors) < len(hop2_neighbors):
                    highlighted_nodes_2.append(hop2_neighbors[frame - len(hop1_neighbors)])
                ax.text(x=0.41,
                        y=0.76,
                        s=f"2nd Hop: {len(highlighted_nodes_2)} neighbors",
                        fontdict={'fontsize': 14, 'fontname': 'Helvetica', 'color': 'yellow'},
                        bbox=dict(facecolor='black', edgecolor='black'))

                for n in self.G.nodes():
                    if n in highlighted_nodes:
                        new_colors.append('orange')
                    elif n in highlighted_nodes_2:
                        new_colors.append('yellow')
                    else:
                        new_colors.append('lightgray')
                new_colors[self.central_node] = 'red'

            node_artists.set_color(new_colors)
            GraphHomophilyAnimation.__draw_formula(ax)
            return node_artists

        # 5. Animate
        ani = animation.FuncAnimation(fig, update, frames=len(hop1_neighbors)*6, interval=self.interval, repeat=False)
        plt.axis('off')
        ax.set_title("Graph Neural Networks: Homophily",
                     x=0.52,
                     y=0.12,
                     fontdict={'size': 21, 'weight': 'bold', 'fontname': 'Helvetica', 'color': 'black'})
        ani.save('homophily_animation.mp4', writer='ffmpeg', fps=self.fps, dpi=300)
        # plt.show()

    """ ----------------------  Private Helper Methods ---------------------  """

    @staticmethod
    def __draw_formula(ax):
        formula = r"$h_{G}=\frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w \in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }$"
        ax.text(x=0.22,
                y=-0.72,
                s=formula,
                horizontalalignment='left',
                fontdict={'fontsize': 13, 'fontweight': 'bold', 'color': 'black'},
                bbox=dict(facecolor='lightgray', edgecolor='black'))

    def __layout(self, ax) -> PathCollection:
        pos = nx.spring_layout(self.G, k=0.5, seed=42)
        nx.draw_networkx_edges(self.G, pos, edge_color='lightgray', ax=ax)
        node_artists = nx.draw_networkx_nodes(self.G, pos, node_color='darkgray', ax=ax)
        nx.draw_networkx_labels(self.G, pos, ax=ax)
        # Highlight central node
        node_colors = ['red' if n == self.central_node else 'lightgray' for n in self.G.nodes()]
        node_artists.set_color(node_colors)
        return node_artists

    def __build_neighbors(self) -> (NodeView, NodeView):
        neighbors = list(self.G.neighbors(self.central_node))
        neighbors_neighbors = []
        for nbr in neighbors:
            nbr_nodes = list(self.G.neighbors(nbr))
            for n in nbr_nodes:
                neighbors_neighbors.append(n)
        neighbors_neighbors = list(set(neighbors_neighbors))
        return neighbors, neighbors_neighbors

    @staticmethod
    def __get_central_node(G: Graph) -> int:
        min_node = None
        half_total_nodes = len(G.nodes)//2

        for node in G.nodes:
            one_hop = set(G.neighbors(node))
            two_hop = set()
            for nbr in one_hop:
                two_hop.update(G.neighbors(nbr))
            total_reach = one_hop | two_hop
            total_reach.discard(node)

            if half_total_nodes - 8 < len(total_reach) < half_total_nodes:
                min_node = node
        return min_node

    @staticmethod
    def __get_starting_node(G: Graph) -> int:
        node_3 = [len(list(G.neighbors(node))) == 1 for node in G.nodes]
        return node_3[0]


if __name__ == '__main__':
    graph_homophily_visualization = GraphHomophilyAnimation(logo_pos=[-0.05, -0.1, 1.1, 1.07],
                                                            interval=1400,
                                                            fps=3,
                                                            num_nodes=96,
                                                            average_degree=0.12)
    graph_homophily_visualization.draw()
