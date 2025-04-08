__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from typing import List
from mlp_layers_vgroup import MLPLayersVGroup


class MLPVGroup(VGroup):
    def __init__(self,
                 layer_sizes: List[int],
                 shift: float,
                 scale: float,
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)

        # Create the layers of neurons with their edges
        layers_group = MLPLayersVGroup.build(layer_sizes)
        self.add_to_back(layers_group)
        # Position the layers
        self.shift(RIGHT*shift)
        self.scale(scale)
