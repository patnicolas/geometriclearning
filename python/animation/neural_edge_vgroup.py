__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from typing import List
from neural_config import NeuralConfig


class NeuralEdgeVGroup(VGroup):
    def __init__(self,
                 layers: List[VMobject],
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)
        self.edge_groups = NeuralEdgeVGroup.__add_edges(layers)

    """ -------------------------   Private helper methods -------------------"""

    @staticmethod
    def __add_edges(layers: List[int]) -> VGroup:
        from itertools import product
        edge_groups = VGroup()
        for l1, l2 in zip(layers[:-1], layers[1:]):
            edge_group = VGroup()
            for n1, n2 in product(l1.neurons, l2.neurons):
                edge = NeuralEdgeVGroup.__get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            edge_groups.add(edge_group)
        return edge_groups

    @staticmethod
    def __get_edge(neuron1, neuron2) -> VMobject:
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
