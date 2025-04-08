__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."
from manim import *
from mlp_vgroup import MLPVGroup
from typing import List
from conv_vgroup import ConvVGroup

"""
"""


class NeuralNetworkScene(ThreeDScene):
    def construct(self):
        tracker = ValueTracker(0)

        conv_group1 = ConvVGroup(4.5, 0.85)
        conv_group2 = ConvVGroup(3.5, 0.70)
        conv_group3 = ConvVGroup(2.5, 0.65)
        layer_sizes = [128, 64, 20, 8]
        num_edge_groups = len(layer_sizes) - 1
        num_edges_per_layer = NeuralNetworkScene.get_num_edges_per_layer(layer_sizes)
        mlp_group = MLPVGroup(layer_sizes, shift=2.0, scale=0.6)
        lines = [line for line in mlp_group.get_family() if isinstance(line, Line)]
        self.play(Rotate(conv_group1, angle=0.44 * PI, axis=UP))
        self.play(Rotate(conv_group2, angle=0.44 * PI, axis=UP))
        self.play(Rotate(conv_group3, angle=0.44 * PI, axis=UP))

        self.play(Write(mlp_group, run_time=0.2, lag_ratio=0.4))

        def edge_color_updater(mob: VMobject):
            current_tracker_value = int(tracker.get_value())
            # Forward pass
            if 1 <= current_tracker_value <= num_edge_groups:
                rel_value = current_tracker_value-1
                state = GREEN
                start_lines = num_edges_per_layer[rel_value]
                end_lines = num_edges_per_layer[rel_value + 1] if rel_value + 1 < len(num_edges_per_layer) else -1
            # Backward pass
            elif num_edge_groups < current_tracker_value <= 2*num_edge_groups:
                rel_value = 2*num_edge_groups - current_tracker_value
                state = RED
                start_lines = num_edges_per_layer[rel_value]
                end_lines = num_edges_per_layer[rel_value + 1] if rel_value + 1 < len(num_edges_per_layer) else -1
            # Idle state
            else:
                start_lines = 0
                end_lines = -1
                state = LIGHT_GRAY

            for line in lines:
                line.set_opacity(0)

            for line in lines[start_lines: end_lines]:
                line.set_color(state)
                line.set_opacity(1)

        mlp_group.add_updater(edge_color_updater)
        
        self.play(tracker.animate.set_value(10), run_time=12)
        self.wait()
        mlp_group.remove_updater(edge_color_updater)

    @staticmethod
    def get_num_edges_per_layer(layer_sizes: List[int]) -> List[int]:
        from itertools import accumulate

        displayed_layer_sizes = [10 if layer_size >= 10 else layer_size for layer_size in layer_sizes]
        source_layer_sizes = displayed_layer_sizes[:-1]
        target_layer_sizes = displayed_layer_sizes[1:]
        num_edges = [0] + [src * tgt for src, tgt in zip(source_layer_sizes, target_layer_sizes)]
        return list(accumulate(num_edges))


if __name__ == '__main__':
    neural_network_scene = NeuralNetworkScene()
    neural_network_scene.construct()
