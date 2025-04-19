from manim import *

from typing import AnyStr, Set, Dict, List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.graph.graph_data_loader import GraphDataLoader


class ManimNeighborLoader(ThreeDScene):
    def construct(self):
        batch = self.get_batch('Flickr')
        neighbor_ids = batch.n_id.tolist()
        hop_1 = set(neighbor_ids)
        hop_1.discard(0)

        entire_scene = VGroup()
        center_dot = Dot(ORIGIN, radius=0.15, color=YELLOW)
        node_objs = self.create_nodes(hop_1, entire_scene, center_dot)

        title = self.display_title(font_sz=36, rt=0.7)

        nodes_creation_label = self.gcn_step(
            r'Sample \ Flickr \ graph \ nodes \ by \ index',
            prev_vm_object=title,
            first=True)
        # Step 1: Show input graph node
        self.play(FadeIn(node_objs[0]))
        self.wait(0.2)

        # Step 2: Show 1-hop neighbors
        for nid in hop_1:
            self.play(FadeIn(node_objs[nid]), run_time=0.1)
        self.wait(0.2)

        # Create and display edges
        edge_objs = self.create_edges(node_objs, hop_1, entire_scene)
        edges_creation_label = self.gcn_step(text=r'Load \ graph \ edges', prev_vm_object=nodes_creation_label)
        self.play(*[Create(e) for e in edge_objs])

        matmul = self.gcn_step(text=r"Message \ passing: \ \  W \cdot h_u^{(l)}", prev_vm_object=edges_creation_label)

        # Step 4: Simulate aggregation (e.g., GCN layer)
        aggregate_arrows = [
            Arrow(
                start=node_objs[nid][0].get_center(),
                end=node_objs[0][0].get_center(),
                color=GREEN,
                buff=0.7,
                stroke_width=8
            )
            for nid in hop_1
        ]
        [entire_scene.add(a) for a in aggregate_arrows]
        self.play(*[GrowArrow(a) for a in aggregate_arrows])
        self.wait(0.2)

        sum_symbol = self.gcn_step(
            text=r"Aggregation: \ s= \sum_{u \in \mathcal{N}(v)} W \cdot h_u^{(l)}",
            prev_vm_object=matmul
        )

        self.play(Rotate(entire_scene, angle=2*PI, axis=UP, about_point=ORIGIN, run_time=3))

        sum_sigma_symbol = self.gcn_step(text="Activation: \ \ \sigma (s)", prev_vm_object=sum_symbol)
        # Step 5: Show updated feature at center node (e.g., color change)
        self.play(center_dot.animate.set_color(RED).scale(2.5))
        self.gcn_step(
            text=r"h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} W \cdot h_u^{(l)} \right)",
            prev_vm_object=sum_sigma_symbol
        )
        self.wait(1)

    # ----------------------------  Helper functions -------------------------------

    # Manage step by step information
    def gcn_step(self, text: AnyStr, prev_vm_object: SVGMobject, first: bool = False) -> SVGMobject:
        vm_object = MathTex(text, font_size=30)
        vm_object.set_color("YELLOW")
        offset = 0.7 if first else 0.2
        vm_object = vm_object.next_to(prev_vm_object, DOWN, buff=offset)
        vm_object.align_to(prev_vm_object, LEFT)
        self.play(Write(vm_object), run_time=0.3)
        self.wait(0.2)
        return vm_object

    # Retrieve batch of nodes and edges
    def get_batch(self, dataset_name: AnyStr) -> np.array:
        # 1.1 Initialize the loader
        graph_data_loader = GraphDataLoader(
            sampling_attributes={
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
        train_data_loader, test_data_loader = graph_data_loader()
        return next(iter(train_data_loader))

    # Create graph nodes
    def create_nodes(self, hop: Set, scene: VGroup, ctr_dot: Dot) -> Dict:
        node_objs = {}
        center_label = Text('0', font_size=16).next_to(ctr_dot, DOWN)
        node_objs[0] = VGroup(ctr_dot, center_label)

        scene.add(node_objs[0])
        node_objs[0].shift(RIGHT * 2)

        angle_step = TAU / len(hop)
        for i, nid in enumerate(hop):
            angle = i * angle_step
            pos = 3 * np.array([np.cos(angle), np.sin(angle), 0])
            dot = Dot(pos, radius=0.15, color=BLUE)
            label = Text(str(nid), font_size=16).next_to(dot, DOWN)
            node_objs[nid] = VGroup(dot, label)

            scene.add(node_objs[nid])
            node_objs[nid].shift(RIGHT * 2)
        return node_objs

    # Create Graph edges
    def create_edges(self, node_objs: Dict, hop_1: Set, scene: VGroup) -> List[VMobject]:
        # Step 3: Draw edges from neighbors to center
        edge_objs = []
        for nid in hop_1:
            edge = Line(
                node_objs[nid][0].get_center(),
                node_objs[0][0].get_center(),
                color=GRAY,
                stroke_width=2
            )
            scene.add(edge)
            edge_objs.append(edge)
        return edge_objs

    def display_title(self, font_sz: int, rt: float) -> SVGMobject:
        title = Tex('Graph Convolutional Neural Network', font_size=font_sz)
        title.to_edge(UL)
        self.play(Write(title), run_time=rt)

        author = Tex('Patrick Nicolas', font_size=font_sz)
        author.to_edge(UR)
        self.play(Write(author), run_time=rt)
        return title


