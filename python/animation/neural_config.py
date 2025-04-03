__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *

NeuralConfig = {
    "neuron_radius": 0.25,
    "neuron_to_neuron_buff": MED_SMALL_BUFF,
    "layer_to_layer_buff": 3,
    "output_neuron_color": BLUE_A,
    "input_neuron_color": WHITE,
    "hidden_layer_neuron_color": YELLOW,
    "neuron_stroke_width": 4,
    "neuron_fill_color": WHITE,
    "edge_color": LIGHT_GREY,
    "edge_stroke_width": 1,
    "edge_propagation_color": YELLOW,
    "edge_propagation_time": 1,
    "max_shown_neurons": 10,
    "brace_for_large_layers": False,
    "average_shown_activation_of_large_layer": True,
    "include_output_labels": False,
    "arrow": True,
    "arrow_tip_size": 0.1,
    "left_size": 1,
    "neuron_fill_opacity": 1
}
