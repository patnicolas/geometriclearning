__author__ = "Patrick Nicolas"
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
from typing import List, Optional
from neural_config import NeuralConfig
from enum import Enum



class LayerType(Enum):
    INPUT = 'Input'
    HIDDEN = 'Hidden'
    OUTPUT = 'Output'

"""
Class/VGroup that defines a layer of neurons/units.
A layer can be INPUT, HIDDEN or OUTPUT
It contains
- List of nodes => VGroup
- Math tex on number of units and other descriptors
- Dots if the number of neurons is large
- Labels for variables for only input and output layers
"""

class MLPLayerVGroup(VGroup):
    def __init__(self,
                 layer_size: int,
                 layer_type: LayerType,
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)
        self.tracker = ValueTracker(0)
        self.layer_type = layer_type
        neurons, math_text, dots = MLPLayerVGroup.__get_layer(layer_size, layer_type)
        self.neurons = neurons
        self.add(neurons)
        self.add(math_text)
        if dots is not None:
            self.add(dots)
        labels = MLPLayerVGroup.__add_labels(neurons, layer_type)
        if labels is not None:
            self.add(labels)

    @staticmethod
    def __add_labels(neurons: VGroup, layer_type: LayerType) -> Optional[VGroup]:
        match layer_type:
            case LayerType.INPUT:
                return MLPLayerVGroup.__label_inputs(neurons)
            case LayerType.OUTPUT:
                return MLPLayerVGroup.__label_outputs(neurons)
            case _:
                return None

    @staticmethod
    def __label_inputs(neurons: VGroup) -> Optional[VGroup]:
        input_labels = VGroup()
        overflow = len(neurons) > NeuralConfig['max_shown_neurons']
        half_display_point = NeuralConfig['max_shown_neurons'] // 2

        for n, neuron in enumerate(neurons):
            index = n + 1
            if overflow and n >= half_display_point:
                index += len(neurons) - NeuralConfig['max_shown_neurons']

            label = MathTex(rf"{{x}}_{{{index}}}", font_size=60)
            label.next_to(neuron, LEFT, buff=0.2)
            input_labels.add(label)
        return input_labels

    @staticmethod
    def __label_outputs(neurons: VGroup):
        output_labels = VGroup()
        for n, neuron in enumerate(neurons):
            index = n + 1
            label = MathTex(rf"\hat{{y}}_{{{index}}}", font_size=60)
            label.next_to(neuron, RIGHT, buff=0.2)
            output_labels.add(label)
        return output_labels

    @staticmethod
    def __get_layer(size: int, layer_type: LayerType) -> (VGroup, MathTex, Tex):
        n_neurons = size
        if n_neurons > NeuralConfig['max_shown_neurons']:
            n_neurons = NeuralConfig['max_shown_neurons']

        neuron_color = MLPLayerVGroup.__get_nn_fill_color(layer_type)
        neurons = VGroup(*[
            Sphere(radius=NeuralConfig['neuron_radius'])
            for _ in range(n_neurons)
        ])
        for neuron in neurons:
            neuron.set_fill(neuron_color)

        neurons.arrange_submobjects(
            DOWN, buff=NeuralConfig['neuron_to_neuron_buff']
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()

        activation = "Softmax" if layer_type == LayerType.OUTPUT else "ReLU"
        text = MathTex(rf'\textbf{{{size}}} \ units \\ \textbf{{{activation}', font_size=48)
        text.next_to(neurons[-1], DOWN, buff=0.6)

        if size > n_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
        else:
            dots = None
        return neurons, text, dots

    @staticmethod
    def __get_nn_fill_color(layer_type: LayerType) -> VMobject:
        match layer_type:
            case LayerType.INPUT:
                return NeuralConfig['input_neuron_color']
            case LayerType.OUTPUT:
                return NeuralConfig['output_neuron_color']
            case LayerType.HIDDEN:
                return NeuralConfig['hidden_layer_neuron_color']