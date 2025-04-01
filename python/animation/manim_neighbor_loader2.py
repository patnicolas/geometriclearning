from click import clear
from manim import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.graph.graph_data_loader import GraphDataLoader



class ManimNeighborLoader2(Scene):
    def construct(self):

        def gcn_step(vm_object: SVGMobject, prev_vm_object: SVGMobject) -> SVGMobject:
            vm_object = vm_object.next_to(prev_vm_object, DOWN, buff=0.2)
            vm_object.align_to(prev_vm_object, LEFT)
            self.play(Write(vm_object))
            self.wait(0.5)
            return vm_object

        # Step 1: Create a loder
        dataset_name = 'Flickr'
        # 1.1 Initialize the loader
        graph_data_loader = GraphDataLoader(
            loader_attributes={
                'id': 'NeighborLoader',
                'num_neighbors': [8, 4, 2],
                'replace': True,
                'batch_size': 18,
                'num_workers': 1
            },
            dataset_name=dataset_name,
            num_subgraph_nodes=2048,
            start_index=70429
        )
        print(graph_data_loader.data)
        train_data_loader, test_data_loader = graph_data_loader()
        batch = next(iter(train_data_loader))

        # Extract node IDs involved in the batch
        center_node = 0
        neighbor_ids = batch.n_id.tolist()
        hop_1 = set(neighbor_ids)
        hop_1.discard(center_node)

        # Layout nodes in a circle around center
        node_objs = {}
        center_dot = Dot(ORIGIN, radius=0.15, color=YELLOW)
        center_label = Text(str(center_node), font_size=18).next_to(center_dot, DOWN)
        node_objs[center_node] = VGroup(center_dot, center_label)
        node_objs[center_node].shift(RIGHT*2)

        angle_step = TAU / len(hop_1)
        for i, nid in enumerate(hop_1):
            angle = i * angle_step
            pos = 3 * np.array([np.cos(angle), np.sin(angle), 0])
            dot = Dot(pos, radius=0.15, color=BLUE)
            label = Text(str(nid), font_size=18).next_to(dot, DOWN)
            node_objs[nid] = VGroup(dot, label)
            node_objs[nid].shift(RIGHT*2)

        title = Tex('Graph Convolutional Neural Network', font_size=36)
        title.to_edge(UL)
        self.play(Write(title))

        nodes_creation_label = gcn_step(Tex('Sample graph nodes by index', font_size=30), title)
        # Step 1: Show input graph node
        self.play(FadeIn(node_objs[center_node]))
        self.wait(0.5)

        # Step 2: Show 1-hop neighbors
        for nid in hop_1:
            self.play(FadeIn(node_objs[nid]), run_time=0.2)
        self.wait(0.2)

        # Step 3: Draw edges from neighbors to center
        edge_objs = []
        for nid in hop_1:
            edge = Line(
                node_objs[nid][0].get_center(),
                node_objs[center_node][0].get_center(),
                color=GRAY,
                stroke_width=2
            )
            edge_objs.append(edge)

        edges_creation_label = gcn_step(Tex('Load graph edges', font_size=30), nodes_creation_label)

        self.play(*[Create(e) for e in edge_objs])
        self.wait(0.5)
        matmul = gcn_step(
            MathTex(r"Matrix \ multiplication \ \  W h_u^{(l)}", font_size=30),
            edges_creation_label
        )

        # Step 4: Simulate aggregation (e.g., GCN layer)
        aggregate_arrows = [
            Arrow(
                start=node_objs[nid][0].get_center(),
                end=node_objs[center_node][0].get_center(),
                color=GREEN,
                buff=0.7,
                stroke_width=8
            )
            for nid in hop_1
        ]
        self.play(*[GrowArrow(a) for a in aggregate_arrows])
        self.wait(1)

        sum_symbol = gcn_step(
            MathTex(r"Aggregation \ \ \sum_{u \in \mathcal{N}(v)} W h_u^{(l)}", font_size=30),
            matmul
        )

        sum_sigma_symbol = gcn_step(
            MathTex(
                r"Activation \ \ \sigma\left( \sum_{u \in \mathcal{N}(v)} W h_u^{(l)} \right)",
                font_size=30
            ),
            sum_symbol
        )

        gcn_step(
            MathTex(
                r"h_v^{(l+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} W h_u^{(l)} \right)",
                font_size=30
            ),
            sum_sigma_symbol
        )

        # Step 5: Show updated feature at center node (e.g., color change)
        self.play(center_dot.animate.set_color(RED).scale(2.0))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
