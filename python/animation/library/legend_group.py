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

from manim import *
from typing import Any, List
from animation.library import colors

class LegendGroup(VGroup):
    def __init__(self, legend_labels: List[MathTex], corner: Any, shift: Any) -> None:
        super(LegendGroup, self).__init__()

        entries = [VGroup(Line(LEFT, RIGHT, color=colors[idx]), legend_label)
                   for idx, legend_label in enumerate(legend_labels)]
        for entry in entries:
            entry.arrange(RIGHT, buff=0.2)
        self.legend = VGroup(entries).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        self.legend.to_corner(corner).shift(shift)
        box = SurroundingRectangle(self.legend, color=DARK_GREY, buff=0.2, fill_opacity=0.2, fill_color=BLACK)
        self.add(self.legend, box)
