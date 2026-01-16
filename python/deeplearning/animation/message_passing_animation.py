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
from util.base_animation import BaseAnimation
from typing import List, AnyStr, Tuple, Dict
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from networkx import Graph
import networkx as nx

class MessagePassingAnimation(BaseAnimation):
    def __init__(self,
                 logo_pos: List[float],
                 interval: int,
                 fps: int,
                 num_features: int,
                 edge_indices: List[Tuple[int, int]]) -> None:
        """
            Default constructor for the animation of Message Passing and aggregation for a Graph Neural Network

            @param logo_pos: Define the position of the chart [x, y, width, height]
            @type logo_pos: List[float]
            @param interval: Interval in milliseconds between frames
            @type interval: int
            @param fps: Number of frame per seconds for animation
            @type fps: int
            @param num_features: Number of features or dimension of node embedding
            @type num_features: int
            @param edge_indices: Definition of edges
            @type edge_indices: List of tuple (source node index -> target node index)
        """
        super(MessagePassingAnimation, self).__init__(logo_pos, interval, fps)
        self.num_features = num_features
        self.edge_indices = edge_indices
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.ax.set_axis_off()

    @abstractmethod
    def _group_name(self) -> AnyStr:
        raise NotImplementedError('_group_name is not implemented')

    def draw(self, mp4_file: bool = False) -> None:
        """
        Draw and animate message passing ana aggregation in a Graph Neural Network. The animation is driven by
        MatplotlibFuncAnimation class that require an update nested function.

        @param mp4_file: Flag to specify if the mp4 file is to be generated (False plot are displayed but not saved)
        @type mp4_file: boolean
        """
        G = nx.Graph()
        G.add_edges_from(self.edge_indices)

        # Initial node features: Array of num_features of random values
        features = {i: np.random.rand(self.num_features) for i in G.nodes()}
        aggregated_features = features.copy()

        # Position of nodes
        pos = nx.spring_layout(G, dim=2, seed=42, scale=3.0)

        def update(frame):
            self.ax.clear()
            self.__title_and_footer(frame)
            self.__draw_graph(G, pos)

            if frame < len(G.nodes()):
                messages = []

                for neighbor in G.neighbors(frame):
                    messages.append(features[neighbor])
                    # Draw the "message" as an arrow
                    x0, y0 = pos[neighbor]
                    x1, y1 = pos[frame]
                    self.ax.annotate("",
                                xy=(x1, y1),
                                xycoords='data',
                                xytext=(x0, y0),
                                textcoords='data',
                                arrowprops=dict(arrowstyle="->", color='red', lw=4))

                # Step 3: Aggregate (sum) messages and update target feature
                if messages:
                    aggregated_features[frame] = sum(messages)

            self.__draw_features_aggregation(G, pos, aggregated_features, frame)
            ani = FuncAnimation(self.fig, update, frames=len(G.nodes()), interval=self.interval, repeat=False)
            ani.save('message_passing.mp4', writer='ffmpeg', fps=self.fps, dpi=240)

    """ ----------------------  Private Helper Methods ---------------------------- """
    
    def __draw_features_aggregation(self,
                                    G: Graph,
                                    pos: Dict[int, List[float]],
                                    aggregated_features: Dict[int, np.array],
                                    target: int) -> None:
        for node in G.nodes():
            f = aggregated_features[node]
            f_str = f'{f[0]:.2f}\n{f[1]:.2f}\n{f[2]:.2f}'
            if target == node:
                color = 'red'
                desc = f'Sum\n'
                font_weight = 'bold'
                font_size = 11
                y_offset = 0.25
            else:
                color = 'black'
                desc = ''
                font_weight = 'regular'
                font_size = 10
                y_offset = 0.2

            self.ax.text(x=pos[node][0] + 0.25,
                         y=pos[node][1] - y_offset,
                         s=f'{desc}{f_str}',
                         fontsize=font_size,
                         ha='center',
                         fontweight=font_weight,
                         color=color)
        self.ax.set_axis_off()

    def __draw_graph(self, G: Graph, pos: Dict[int, List[float]]) -> None:
        x_shadow_offset = 0.03
        y_shadow_offset = 0.056

        # Shadow for nice 3D look and feel
        shadow_pos = {k: [v[0] + x_shadow_offset, v[1] + y_shadow_offset] for k, v in pos.items()}
        _ = nx.draw_networkx_nodes(G,
                                   shadow_pos,
                                   node_size=1200,
                                   ax=self.ax,
                                   node_color='grey',
                                   edgecolors='grey',
                                   linewidths=0)
        # Draw the graph nodes
        _ = nx.draw_networkx_nodes(G,
                                   pos,
                                   node_size=1200,
                                   ax=self.ax,
                                   node_color='blue',
                                   edgecolors='black',
                                   linewidths=1)
        # Draw Graph edges
        nx.draw_networkx_edges(G, pos)
        # Draw Graph node labels
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=14)

    def __title_and_footer(self, frame: int) -> None:
        # Footer
        self.ax.text(x=-1.75,
                     y=-3.6,
                     s='Hands-on Geometric Deep Learning',
                     ha='left',
                     color='blue',
                     fontdict={'fontsize': 18, 'fontname': 'Helvetica'})

        # The title is repeated with a slight offset (0.01) to give a 3D look and feel
        self.ax.text(x=-1.01,
                     y=-0.651,
                     s="Graph Neural Network\nMessage Passing Tuning\nStep {}".format(frame + 1),
                     ha='center',
                     color='grey',
                     fontdict={'fontsize': 22, 'fontname': 'Helvetica'})
        self.ax.text(x=-1.0,
                     y=-0.65,
                     s="Graph Neural Network\nMessage Passing Tuning\nStep {}".format(frame + 1),
                     ha='center',
                     color='black',
                     fontdict={'fontsize': 22, 'fontname': 'Helvetica'})


if __name__ == '__main__':
    edges = [(0, 1), (0, 2), (0, 3), (0, 11), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 16), (3, 16), (4, 12),
             (4, 13), (6, 14), (6, 15), (5, 16), (1, 10)]
    message_passing_animation = MessagePassingAnimation(logo_pos=[0.015, 0.725, 0.3, 0.28],
                                                        interval=800,
                                                        fps=2,
                                                        num_features=3,
                                                        edge_indices=edges)
    message_passing_animation.draw(mp4_file=True)
