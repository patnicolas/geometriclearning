__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from typing import List
from neural_layer_vgroup import NeuralLayerVGroup
from neural_edge_vgroup import NeuralEdgeVGroup
from neural_label_vgroup import NeuralLabelVGroup


class NeuralNetworkVGroup(VGroup):
    def __init__(self, layer_sizes: List[int], *args, **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)

        neural_layer_vgroup = NeuralLayerVGroup(layer_sizes)
        self.layers = neural_layer_vgroup.layers

        neural_edge_vgroup = NeuralEdgeVGroup(self.layers)
        self.add_to_back(neural_edge_vgroup.edge_groups)
        self.add_to_back(self.layers)

        neural_edge_group = NeuralLabelVGroup(self.layers,  layer_sizes)
        self.add(neural_edge_group.labels)
