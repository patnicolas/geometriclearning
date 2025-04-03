__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from neural_network_vgroup import NeuralNetworkVGroup


class NeuralNetworkScene(Scene):
    def construct(self):
        manim_neural_network_group = NeuralNetworkVGroup([1000, 32, 16, 8])

        manim_neural_network_group.scale(0.7)
        self.play(Write(manim_neural_network_group, run_time=3, lag_ratio=0.6))
        self.wait()
