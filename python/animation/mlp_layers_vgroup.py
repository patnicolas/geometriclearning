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
from typing import List, Self
from neural_config import NeuralConfig
from mlp_layer_vgroup import MLPLayerVGroup, LayerType


class MLPLayersVGroup(VGroup):
    def __init__(self,
                 layers: List[MLPLayerVGroup],
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)
        layers_group = VGroup(*[layers])
        layers_group.arrange_submobjects(RIGHT, buff=NeuralConfig['layer_to_layer_buff'])
        self.add(layers_group)
        edges = MLPLayersVGroup.__add_edges(layers)
        self.add_to_back(edges)

    @classmethod
    def build(cls, layer_sizes: List[int]) -> Self:
        num_layers = len(layer_sizes)
        return cls([MLPLayerVGroup(layer_size, MLPLayersVGroup.__get_nn_fill_color(idx, num_layers))
                    for idx, layer_size in enumerate(layer_sizes)])

    """ ------------------------------  Private Helper Methods ---------------  """

    @staticmethod
    def __add_edges(layers: List[MLPLayerVGroup]) -> VGroup:
        edges_group = VGroup()
        source_layers = layers[:-1]
        target_layers = layers[1:]
        for l1, l2 in zip(source_layers, target_layers):
            edge_group = MLPLayersVGroup.__add_edge_group(l1, l2)
            edges_group.add(edge_group)
        return edges_group

    @staticmethod
    def __add_edge_group(in_layer: MLPLayerVGroup, out_layer: MLPLayerVGroup) -> VGroup:
        from itertools import product

        all_in_children = in_layer.get_family()
        all_out_children = out_layer.get_family()
        edge_group = VGroup()
        in_neurons = [m for m in all_in_children if isinstance(m, Sphere)]
        out_neurons = [m for m in all_out_children if isinstance(m, Sphere)]

        for n1, n2 in product(in_neurons, out_neurons):
            edge = MLPLayersVGroup.__get_edge(n1, n2)
            edge_group.add(edge)
            n1.edges_out.add(edge)
            n2.edges_in.add(edge)
        return edge_group

    @staticmethod
    def __get_edge(neuron1: VGroup, neuron2: VGroup) -> VMobject:
        if NeuralConfig['arrow']:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=NeuralConfig['neuron_radius'],
                stroke_color=NeuralConfig['edge_color'],
                stroke_width=NeuralConfig['edge_stroke_width'],
                tip_length=NeuralConfig['arrow_tip_size']
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=NeuralConfig['neuron_radius'],
            stroke_color=NeuralConfig['edge_color'],
            stroke_width=NeuralConfig['edge_stroke_width']
        )

    @staticmethod
    def __get_nn_fill_color(index: int, num_layers: int) -> LayerType:
        if index >= num_layers - 1:
            index = -1
        match index:
            case 0:
                return LayerType.INPUT
            case -1:
                return LayerType.OUTPUT
            case _:
                return LayerType.HIDDEN
