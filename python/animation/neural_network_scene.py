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
        mlp_tracker = ValueTracker(0.0)
        title = Tex(r'\textbf{Hands-on Geometric Deep Learning}', font_size=42, color=WHITE).to_edge(UP)
        self.add(title)

        def description(conv_layer_vgroup: VGroup):
            conv_layer_text = Tex(r'\textbf{Convolutional \\ layer}', font_size=20)
            conv_layer_text.next_to(conv_layer_vgroup, DOWN, buff=0.5).shift(LEFT)

            start_point: np.array = conv_layer_vgroup.get_center()
            x = start_point[0]
            y = start_point[1]
            z = start_point[2]

            line = Line(
                np.array([x + 0.25, y - 0.4, z]),
                np.array([x + 0.25, y - 2.5, z]),
                buff=0.1,
                stroke_color=YELLOW,
                stroke_width=4
            )
            receptive_fields = Tex(r'\textbf{Receptive \\ fields}', font_size=20, color=YELLOW)
            receptive_fields.next_to(line, DOWN, buff=0.1)
            self.play(Write(conv_layer_text), Write(line), Write(receptive_fields), run_time=0.4)

        # -------------  Convolution layers ------------------
        conv_group1 = ConvVGroup(shift=3.2, scale=0.85, opacity=0.2)
        conv_group2 = ConvVGroup(shift=2.2, scale=0.70, opacity=0.2)
        conv_group3 = ConvVGroup(shift=1.5, scale=0.50, opacity=0.2)

        self.play(Rotate(conv_group1, angle=0.5 * PI, axis=UP),
                  Rotate(conv_group2, angle=0.5 * PI, axis=UP),
                  Rotate(conv_group3, angle=0.5 * PI, axis=UP),
                  run_time=0.5)
        description(conv_group1)
        conv_group1.set_opacity(1.0)
        conv_group2.set_opacity(1.0)
        conv_group3.set_opacity(1.0)

        layer_sizes = [1764, 128, 32, 8]
        num_edge_groups = len(layer_sizes) - 1
        num_edges_per_layer = NeuralNetworkScene.get_num_edges_per_layer(layer_sizes)
        mlp_group = MLPVGroup(layer_sizes, shift=2.0, scale=0.45)
        self.play(Write(mlp_group, run_time=0.1))
        lines = [line for line in mlp_group.get_family() if isinstance(line, Line)]

        def mlp_color_updater(mob: Mobject):
            current_tracker_value = int(mlp_tracker.get_value())

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
            elif 2*num_edge_groups <= current_tracker_value < 2*num_edge_groups+6:
                state = RED
                start_lines = num_edges_per_layer[0]
                end_lines = num_edges_per_layer[1]
            else:
                start_lines = 0
                end_lines = -1
                state = LIGHT_GRAY

            for line in lines:
                line.set_opacity(0)
            for line in lines[start_lines: end_lines]:
                line.set_color(state)
                line.set_opacity(1)

        status_text_forward = Tex(r'\textbf{Forward Weights Propagation}', font_size=32, color=GREEN).to_edge(DOWN)
        self.add(status_text_forward)
        mlp_group.add_updater(mlp_color_updater)
        status_text_backward = Tex(r'\textbf{Backward Loss Gradient Propagation}', font_size=32, color=RED).to_edge(DOWN)
        # self.add(status_text_backward)
        self.play(
            mlp_tracker.animate.set_value(16),
            ReplacementTransform(status_text_forward, status_text_backward),
            run_time=16)

        conv_group1.set_opacity(0.2)
        conv_group2.set_opacity(0.2)
        conv_group3.set_opacity(0.2)
        self.wait()

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
