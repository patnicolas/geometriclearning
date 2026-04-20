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
import math
from gauge_group import GaugeGroup, GaugeScene, GaugeConfig
from scatter_2d_dynamic_group import Scatter2DConfig, Scatter2DDynamicScene, Scatter2DDynamicGroup
from dataclasses import dataclass
from typing import AnyStr, Tuple, List


@dataclass
class ComposerConfig:
    title: AnyStr
    axis_font_size: int
    display_scale: Tuple[float, float]
    run_time: int
    scatter_2d_config: Scatter2DConfig
    gauge_config: GaugeConfig


class ComposerScene(Scene):
    data_points = [(x, math.exp(0.5*x)) for x in range(20)]

    def construct(self):
        # Setup Value tracker
        vt = ValueTracker(0)
        # Retrieve the configuration
        composer_config = ComposerScene.__get_config()
        # Add Gauge
        gauge_group = self.__add_gauge(vt, composer_config)
        # Add Scatter 2D group located
        scatter_2d_group = self.__add_scatter_2d_group(vt, composer_config, gauge_group)
        self.play(vt.animate.set_value(len(ComposerScene.data_points) - 1),
                  run_time=composer_config.run_time,
                  rate_func=linear)
        self.wait()

    """ -------------------  Private Helper Methods  ------------------- """

    def __add_gauge(self, vt: ValueTracker, composer_config: ComposerConfig) -> GaugeGroup:
        y = [item[1] for item in ComposerScene.data_points]
        gauge_group = GaugeGroup(vt=vt,
                                 gauge_config=composer_config.gauge_config,
                                 data_points=y).to_edge(UR)
        gauge_group.add_updater(gauge_group.get_updater(vt))
        gauge_group.scale(composer_config.display_scale[0])
        self.add(gauge_group)
        return gauge_group

    def __add_scatter_2d_group(self,
                               vt: ValueTracker,
                               composer_config: ComposerConfig,
                               gauge_group: GaugeGroup):
        scatter_2d_group = Scatter2DDynamicGroup(scatter_2d_config=composer_config.scatter_2d_config,
                                                 data_points=ComposerScene.data_points).to_edge(DR)
        scatter_2d_group.add_updater(scatter_2d_group.get_updater(vt))
        scatter_2d_group.scale(composer_config.display_scale[1])
        self.add(scatter_2d_group)
        scatter_2d_group.next_to(gauge_group, DOWN, buff=0.5)
        return scatter_2d_group

    @staticmethod
    def __get_config() -> ComposerConfig:
        title = r'Widget \ Composer'
        axis_font_size = 14
        display_scale = (0.8, 0.8)

        scatter_2d_config = Scatter2DConfig(xy_labels=("X", "Y"),
                                            lengths=(4, 4),
                                            title=MathTex(title, font_size=32),
                                            num_lines=5,
                                            radius=0.1,
                                            label_font_size=28,
                                            axis_font_size=axis_font_size+4,
                                            legend_texts=('Value',))
        gauge_config = GaugeConfig(radius=2.5, num_ticks=11, font_size=axis_font_size)
        return ComposerConfig(title=title,
                              axis_font_size=axis_font_size,
                              display_scale=display_scale,
                              run_time=4,
                              scatter_2d_config=scatter_2d_config,
                              gauge_config=gauge_config)

if __name__ == '__main__':
    scene = ComposerScene()
    scene.construct()

