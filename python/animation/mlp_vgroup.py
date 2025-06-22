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
