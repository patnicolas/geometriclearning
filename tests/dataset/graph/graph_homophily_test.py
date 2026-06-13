import unittest
import logging
from dataset.graph.graph_homophily import GraphHomophily, GraphHomophilyType
from torch_geometric.data import Data
import torch
import os
import python
from python import SKIP_REASON


class GraphHomophilyTest(unittest.TestCase):

    def test_init_1(self):
        try:
            homophily_flickr = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.Node)
            logging.info(homophily_flickr)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_2(self):
        try:
            homophily_cora = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.Node)
            logging.info(homophily_cora)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_homegrown_edge_homophily(self):
        try:
            labels = torch.Tensor([0, 0, 1, 1, 0])
            data = Data(y=labels,
                        edge_index=torch.tensor([
                                            [0, 1, 2, 3],
                                            [1, 2, 3, 4]]),
                        num_nodes=len(labels)
                        )
            homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Edge)
            edge_homophily = homophily.compute()
            logging.info(f'Edge homophily: {edge_homophily}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_homegrown_node_homophily(self):
        try:
            labels = torch.Tensor([0, 0, 1, 1, 0])
            data = Data(y=labels,
                        edge_index=torch.tensor([
                                            [0, 1, 2, 3],
                                            [1, 2, 3, 4]]),
                        num_nodes=len(labels)
                        )
            homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Node)
            node_homophily = homophily.compute()
            logging.info(f'Node homophily: {node_homophily}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_edge_homophily(self):
        try:
            labels = torch.Tensor([0, 0, 1, 1, 0])
            data = Data(y=labels,
                        edge_index=torch.tensor([
                                            [0, 1, 2, 3],
                                            [1, 2, 3, 4]]),
                        num_nodes=len(labels)
                        )
            homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Edge)
            edge_homophily = homophily()
            logging.info(f'Edge homophily: {edge_homophily}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_node_homophily(self):
        try:
            labels = torch.Tensor([0, 0, 1, 1, 0])
            data = Data(y=labels,
                        edge_index=torch.tensor([
                                            [0, 1, 2, 3],
                                            [1, 2, 3, 4]]),
                        num_nodes=len(labels)
                        )
            homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Node)
            node_homophily = homophily()
            logging.info(f'Node homophily: {node_homophily}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_node_homophily_datasets(self):
        try:
            homophily = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.Node)
            node_homophily = homophily()
            logging.info(f'Flickr node homophily: {node_homophily:.3f}')

            homophily = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.Node)
            node_homophily = homophily()
            logging.info(f'Cora node homophily: {node_homophily:.3f}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_edge_homophily_datasets(self):
        try:
            homophily = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.Edge)
            edge_homophily = homophily()
            logging.info(f'Flickr edge homophily: {edge_homophily:.3f}')

            homophily = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.Edge)
            edge_homophily = homophily()
            logging.info(f'Cora edge homophily: {edge_homophily:.3f}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_class_insensitive_edge_homophily_datasets(self):
        try:
            homophily = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.ClassInsensitiveEdge)
            edge_homophily = homophily()
            logging.info(f'Flickr class insensitive edge homophily: {edge_homophily:.3f}')

            homophily = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.ClassInsensitiveEdge)
            edge_homophily = homophily()
            logging.info(f'Cora class insensitive edge homophily: {edge_homophily:.3f}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_all(self):
        try:
            dataset_names = ['Cora', 'PubMed', 'CiteSeer', 'Wikipedia', 'Flickr']
            homophily_types = [GraphHomophilyType.Node, GraphHomophilyType.Edge, GraphHomophilyType.ClassInsensitiveEdge]
            for dataset_name in dataset_names:
                for homophily_type in homophily_types:
                    homophily = GraphHomophily.build(dataset_name=dataset_name, homophily_type=homophily_type)
                    homophily_factor = homophily()
                    logging.info(f'{dataset_name} {homophily_type.value} homophily: {homophily_factor:.3f}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_message_propagation(self):
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
        node_colors = ['red' if n == central_node else 'lightgray' for n in G.featured_nodes()]
        node_artists.set_color(node_colors)

        # 4. Animation update function
        highlighted_nodes = [central_node]
        highlighted_nodes_2 = [central_node]

        def update(frame):
            new_colors = []
            if frame < len(neighbors):
                highlighted_nodes.append(neighbors[frame])
                new_colors = ['orange' if n in highlighted_nodes else 'lightgray' for n in G.featured_nodes()]
                new_colors[central_node] = 'red'
            else:
                # DEBUG
                highlighted_nodes_2.append(neighbors_neighbors[frame- len(neighbors)])
                for n in G.featured_nodes():
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
        ani = animation.FuncAnimation(fig, update, frames=len(neighbors)*4, interval=800, repeat=False)
        plt.axis('off')
        logging.info('show')
        plt.show()
