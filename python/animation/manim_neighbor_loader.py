from manim import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.graph.graph_data_loader import GraphDataLoader
from typing import Dict, AnyStr, List


class ManimNeighborLoader(Scene):
    def construct(self):
        # Step 1: Create a loder
        dataset_name = 'Flickr'
        # 1.1 Initialize the loader
        graph_data_loader = GraphDataLoader(
            loader_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': [8, 4, 2],
                'replace': True,
                'batch_size': 16,
                'num_workers': 1
            },
            dataset_name=dataset_name,
            num_subgraph_nodes=2048)
        print(graph_data_loader.data)

        # 1.2 Extract the loader for training and validation sets
        train_data_loader, test_data_loader = graph_data_loader()
        batch = next(iter(train_data_loader))

        # Step 2: Create nodes and edges
        def create_nodes(neighbor_ids: List[AnyStr]) -> Dict[int, VGroup]:
            node_objs = {}
            center_node = 0
            for i, nid in enumerate(neighbor_ids):
                angle = i * TAU / len(neighbor_ids)
                x = 3 * np.cos(angle)
                y = 3 * np.sin(angle)
                dot = Dot(point=np.array([x, y, 0]), radius=0.15, color=GRAY)
                label = Text(str(nid), font_size=16).next_to(dot, DOWN)
                node_objs[nid] = VGroup(dot, label)

            center_dot = Dot(point=ORIGIN, radius=0.15, color=YELLOW)
            center_label = Text(str(center_node), font_size=16).next_to(center_dot, DOWN)
            node_objs[center_node] = VGroup(center_dot, center_label)
            return node_objs

        def create_edges(node_objs: Dict[int, VGroup]) -> List[Line]:
            edges = []
            for src, dst in graph_data_loader.data.edge_index.t().tolist():
                if src in node_objs and dst in node_objs:
                    start = node_objs[src][0].get_center()
                    end = node_objs[dst][0].get_center()
                    edges.append(Line(start, end, stroke_width=2, color=BLUE))
            return edges

        print('Create nodes')
        node_objects = create_nodes(batch.n_id.tolist())
        edge_objects = create_edges(node_objects)
        self.wait(1)

        # Step 3: Animate hops and edges
        def animate_hops(node_objs: Dict[int, VGroup]) -> None:
            self.play(FadeIn(node_objs[0]))
            hop_0 = set(batch.n_id[0: batch.batch_size].tolist())
            hop_1 = set(batch.n_id[batch.batch_size * 1: batch.batch_size * 2].tolist())
            hop_2 = set(batch.n_id[batch.batch_size * 2:].tolist())
            for nid in hop_0:
                self.play(FadeIn(node_objs[nid]), run_time=0.1)
            self.wait(2.0)
            # Animate first-hop neighbors
            for nid in hop_1:
                self.play(FadeIn(node_objs[nid]), run_time=0.1)
            self.wait(2.0)
            # Animate second-hop neighbors
            for nid in hop_2:
                self.play(FadeIn(node_objs[nid]), run_time=0.2)
            self.wait(1)

        def animate_edges(edges: List[Line]) -> None:
            self.play(*[Create(edge) for edge in edges], run_time=2)

        animate_hops(node_objects)
        animate_edges(edge_objects)

        self.wait(2)


if __name__ == '__main__':
    r = ManimNeighborLoader()
    r.construct()
