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


# Python standard library imports
from typing import AnyStr, List
import logging
# Library imports
from play import Play
from dataset.graph.graph_homophily import GraphHomophily, GraphHomophilyType
import python


class GraphHomophilyPlay(Play):
    """
      Source code related to the Substack article 'Neighbors Matter: How Homophily Shapes Graph Neural Networks'.
      Reference: https://patricknicolas.substack.com/p/neighbors-matter-how-homophily-shapes

      Source code for homophily computation:
      https://github.com/patnicolas/geometriclearning/blob/main/python/dataset/graph/graph_homophily.py

      The features are implemented by the class GraphHomophily in the source file
                    python/dataset/graph/graph_homophily.py
      The class GraphHomophilyPlay is a wrapper of the class GraphHomophily
    """
    def __init__(self, dataset_names: List[AnyStr], homophily_types: List[GraphHomophilyType]) -> None:
        super(GraphHomophilyPlay, self).__init__()
        self.dataset_names = dataset_names
        self.homophily_types = homophily_types

    def play(self) -> None:
        self.play_build_homophily()
        GraphHomophilyPlay.play_animation()

    def play_build_homophily(self) -> None:
        """
        Source code related to Substack article 'Neighbors Matter: How Homophily Shapes Graph Neural Networks' -
        Section Node & Edge Homophily - Code snippet 3
        Ref: https://patricknicolas.substack.com/p/neighbors-matter-how-homophily-shapes
        """
        for dataset_name in self.dataset_names:
            for homophily_type in self.homophily_types:
                homophily = GraphHomophily.build(dataset_name=dataset_name, homophily_type=homophily_type)
                homophily_factor = homophily()
                logging.info(f'{dataset_name} {homophily_type.value} homophily: {homophily_factor:.3f}')

    @staticmethod
    def play_animation() -> None:
        """
        Source code related to the animation of the Substack article
            'Neighbors Matter: How Homophily Shapes Graph Neural Networks'.
        Ref:  https://patricknicolas.substack.com/p/neighbors-matter-how-homophily-shapes
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.animation as animation

        # 1. Create the graph
        G = nx.erdos_renyi_graph(n=48, p=0.1, seed=42)
        central_node = 1
        neighbors = list(G.neighbors(central_node))
        neighbors_neighbors = []
        for nbr in neighbors:
            nbr_nodes = list(G.neighbors(nbr))
            logging.info(f'nbr_nodes: {nbr_nodes}')
            for n in nbr_nodes:
                neighbors_neighbors.append(n)
        neighbors_neighbors = list(set(neighbors_neighbors))

        # 2. Layout
        pos = nx.spring_layout(G, seed=42)

        # 3. Setup the figure and draw base graph
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', ax=ax)
        node_artists = nx.draw_networkx_nodes(G, pos, node_color='darkgray', ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)

        # Highlight central node
        node_colors = ['red' if n == central_node else 'lightgray' for n in G.nodes()]
        node_artists.set_color(node_colors)

        # 4. Animation update function
        highlighted_nodes = [central_node]
        highlighted_nodes_2 = [central_node]

        def update(frame):
            new_colors = []
            if frame < len(neighbors):
                highlighted_nodes.append(neighbors[frame])
                new_colors = ['orange' if n in highlighted_nodes else 'lightgray' for n in G.nodes()]
                new_colors[central_node] = 'red'
            else:
                # DEBUG
                highlighted_nodes_2.append(neighbors_neighbors[frame - len(neighbors)])
                for n in G.nodes():
                    if n in highlighted_nodes:
                        new_colors.append('orange')
                    elif n in highlighted_nodes_2:
                        new_colors.append('yellow')
                    else:
                        new_colors.append('lightgray')
                new_colors[central_node] = 'red'

            node_artists.set_color(new_colors)
            return node_artists

        # 5. Animate
        ani = animation.FuncAnimation(fig, update, frames=len(neighbors) * 4, interval=800, repeat=False)
        plt.axis('off')
        logging.info('show')
        plt.show()


if __name__ == "__main__":
    datasets = ['Cora', 'PubMed', 'CiteSeer', 'Wikipedia', 'Flickr']
    homophily_categories = [GraphHomophilyType.Node, GraphHomophilyType.Edge, GraphHomophilyType.ClassInsensitiveEdge]
    graph_homophily_play = GraphHomophilyPlay(datasets, homophily_categories)
    graph_homophily_play.play()
