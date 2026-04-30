__author__ = "Patrick Nicolas"
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

from manim import *
from typing import List, Any, Callable
from networks.mlp_vgrp import MLPVGrp

class MLPNetworkVGrp(VGroup):
    def __init__(self,
                 layer_sizes: List[int],
                 shift: float,
                 scale: float,
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)

        # Create the layers of neurons with their edges
        self.layers_group = MLPVGrp.build(layer_sizes)
        self.add_to_back(self.layers_group)
        # Position the layers
        self.shift(RIGHT*shift)
        self.scale(scale)

    def get_updater(self, vt: ValueTracker) -> Callable[[Any], Any]:
        def updater(mob: Mobject):
            current_tracker_value = int(vt.get_value())
            state = GREEN if current_tracker_value % 2 == 0 else RED
            print(f'Tracking value {current_tracker_value} state = {state}')
            MLPVGrp.set_edges_color(state)
        return updater


class MLPNetworkScene(ThreeDScene):
    def construct(self):
        mlp_tracker = ValueTracker(0.0)
        title = Tex(r'\textbf{Hands-on Geometric Deep Learning}', font_size=42, color=WHITE).to_edge(UP)
        self.add(title)

        layer_sizes = [8, 4]
        # num_edge_groups = len(layer_sizes) - 1
        # num_edges_per_layer = MLPNetworkScene.get_num_edges_per_layer(layer_sizes)
        mlp_group = MLPNetworkVGrp(layer_sizes, shift=2.0, scale=0.45)
        self.play(FadeIn(mlp_group, run_time=0.1))
        # lines = [line for line in mlp_group.get_family() if isinstance(line, Line)]

        status_text_forward = Tex(r'\textbf{Forward Weights Propagation}',
                                  font_size=32,
                                  color=GREEN).to_edge(DOWN)
        self.add(status_text_forward)
        mlp_group.add_updater(mlp_group.get_updater(mlp_tracker))
        status_text_backward = Tex(r'\textbf{Backward Loss Gradient Propagation}',
                                   font_size=32,
                                   color=RED).to_edge(DOWN)
        # self.add(status_text_backward)
        """
        self.play(
            mlp_tracker.animate.set_value(12),
            ReplacementTransform(status_text_forward, status_text_backward),
            run_time=12)
        """
        self.play(mlp_tracker.animate.set_value(12), run_time=12)
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
    scene = MLPNetworkScene()
    scene.construct()

