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

from manim import *

NeuralConfig = {
    "neuron_radius": 0.38,
    "neuron_to_neuron_buff": MED_SMALL_BUFF,
    "layer_to_layer_buff": 2.0,
    "output_neuron_color": RED,
    "input_neuron_color": BLUE,
    "hidden_layer_neuron_color": YELLOW,
    "neuron_stroke_width": 4,
    "neuron_fill_color": BLUE,
    "edge_color": LIGHT_GREY,
    "edge_stroke_width": 1,
    "edge_propagation_color": YELLOW,
    "edge_propagation_time": 1,
    "max_shown_neurons": 10,
    "brace_for_large_layers": False,
    "average_shown_activation_of_large_layer": True,
    "include_output_labels": False,
    "arrow": True,
    "arrow_tip_size": 0.2,
    "left_size": 1,
    "neuron_fill_opacity": 1
}
