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
from animation.library.plots.gauge_vgrp import GaugeVGrp, GaugeConfig
from animation.library.math.sgd_vgrp import SGDVGRP, parabolic_trig_loss



class SGD3DScene(ThreeDScene):

    def construct(self):
        import math

        max_gauge_index = 20

        vt = ValueTracker(0)
        gauge_config = GaugeConfig(radius=2.5, num_ticks=11, font_size=16)
        gauge_group = GaugeVGrp(title_str="Loss",
                                data_points=[100*(3 - math.log(x)) for x in range(1, max_gauge_index)],
                                gauge_config=gauge_config).shift(3 * RIGHT).shift(1.2*DOWN)
        self.add(gauge_group)
        gauge_group.add_updater(gauge_group.get_updater(vt))

        sgd_vgrp = SGDVGRP(title="Stochastic Gradient Langevin Dynamics - Ackley Sampling",
                           optimization_surface=parabolic_trig_loss,
                           noise_level=0.6,
                           num_steps=40,
                           lr=0.25).scale(0.8).shift(-4*OUT).shift(3.5*LEFT)

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(sgd_vgrp)
        self.play(Create(sgd_vgrp))
        self.add_fixed_in_frame_mobjects(sgd_vgrp.title)
        sgd_vgrp.noisy_path.set_opacity(1.0)

        self.play(Create(sgd_vgrp.noisy_path.scale(0.95)).set_run_time(8),
                  vt.animate.set_value(max_gauge_index),
                  run_time=10)
        self.stop_ambient_camera_rotation()
        self.wait(2)


if __name__ == '__main__':
    scene = SGD3DScene()
    scene.construct()
