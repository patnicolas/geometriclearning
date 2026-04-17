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

from typing import Tuple, List
from manim import *
import math

Rnge = Tuple[int, int] | Tuple[int, int, int]

colors = (BLUE, RED, YELLOW, WHITE)

def get_num_ticks(range: List[float]) -> int:
    return int((range[1]-range[0])/range[2])

def extract_scale_factor(x: float) -> float:
    scale = 1e-4
    while x > scale:
        scale *= 10
    return scale*0.1


def get_2d_ranges(data: List[Tuple[float, ...]], num_lines: int) -> Tuple[List[float], List[float]]:
    all_x, all_y = data
    x_max = max(all_x)
    y_max = max(all_y)
    x_min = min(all_x)
    y_min = min(all_y)

    scale = (x_max - x_min)/num_lines
    x_range = [x_min, x_max*1.01, scale]
    scale = (y_max - y_min)/num_lines
    y_range = [y_min, y_max*1.01, scale]
    return x_range, y_range

