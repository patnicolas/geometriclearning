from manim import *
from typing import Callable, Tuple
from animation.library import Rnge
import numpy as np
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


class ParametricSurfaceVGrp(VGroup):
    def __init__(self,
                 func: Callable[[float, float], np.array],
                 u_range: Rnge,
                 v_range: Rnge,
                 scale: float,
                 resolution: Tuple[int, int],
                 title: MathTex,
                 **kwargs) -> None:
        super(ParametricSurfaceVGrp, self).__init__(**kwargs)
        self.func = func
        self.u_range = u_range
        self.v_range = v_range

        self.surface = Surface(func, u_range, v_range, resolution).scale(scale)
        self.title = title
        self.add(title, self.surface)

    def get_attributes(self) -> Tuple[Write, Create]:
        return Write(self.title, run_time=1), Create(self.surface, run_time=4)