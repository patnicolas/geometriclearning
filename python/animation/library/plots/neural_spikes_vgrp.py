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
import numpy as np
from typing import Tuple, List


class SpikeTrainVGrp(VGroup):
    def __init__(self,
                 duration: float,
                 num_neurons: int,
                 spike_probability: float,
                 gauss_sigma: float) -> None:
        super(SpikeTrainVGrp, self).__init__()
        self.duration = duration
        self.num_neurons = num_neurons
        self.spike_probability = spike_probability
        self.gauss_sigma = gauss_sigma

    def __call__(self) -> Tuple[VGroup, ...]:
        axes, labels = self.__create_axes_labels()
        self.add(axes, labels)
        return self.__create_spikes(axes)

    """  ---------------------  Private Supporting Methods  ---------------------  """

    def __create_axes_labels(self) -> Tuple[ThreeDAxes, VGroup]:
        axes = ThreeDAxes(
            x_range=[0, 12, 1],   # x_range: Time (0 to 12 seconds)
            y_range=[0, 9, 1],    # y_range: Neuron Index (1 to 6)
            z_range=[0, 2, 1],           # z_range: Spike Height
            x_length=10,
            y_length=8,
            z_length=2,
            axis_config={"include_tip": True}
        )
        labels = axes.get_axis_labels(
            x_label="Time",
            y_label="Neuron ID",
            z_label="Voltage"
        )
        return axes, labels

    def __create_spikes(self, axes: ThreeDAxes) -> Tuple[VGroup, VGroup]:
        spikes = VGroup()
        smoothed_curves = VGroup()

        for n in range(1, self.num_neurons + 1):
            space = n * 1.5
            # Create a "line" for each neuron to sit on
            neuron_line = Line(
                axes.c2p(0, space, 0),
                axes.c2p(self.duration, space, 0),
                color=GRAY,
                stroke_opacity=0.5
            )
            self.add(neuron_line)

            # Generate random timestamps
            # spike_times = np.arange(0.5, duration, 0.5)
            spike_times = [t for t in np.arange(0.5, self.duration, 0.4)
                           if np.random.random() < self.spike_probability]

            for t in spike_times:
                spike = Line(
                    axes.c2p(t, space, 0),
                    axes.c2p(t, space, 1),
                    color=YELLOW,
                    stroke_width=4
                )
                spikes.add(spike)

            # Define the smoothing function (Sum of Gaussians)
            # F(t) = sum( exp( -(t - spike_time)^2 / (2 * sigma^2) ) )
            def gaussian_smoothing_rate(t, times: List[float], s: float):
                return sum(np.exp(-(t - st) ** 2 / (2 * s ** 2)) for st in times) if times else 0

            # Create the smooth curve trace
            # We use ParametricFunction to place it at the correct 'Neuron ID' (y-axis)
            smooth_curve = axes.plot_parametric_curve(
                lambda t: np.array([t, space, gaussian_smoothing_rate(t,  spike_times, self.gauss_sigma)]),
                t_range=[0, self.duration],
                color=RED  # BLUE_A
            )
            smoothed_curves.add(smooth_curve)
        return spikes, smoothed_curves


class SpikeTrainScene(ThreeDScene):
    def construct(self):
        spike_train_vgrp = SpikeTrainVGrp(duration=12, num_neurons=6,spike_probability=0.4, gauss_sigma=0.19)
        self.add(spike_train_vgrp)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        spikes, smoothed_curves = spike_train_vgrp()

        self.play(Create(spikes), Create(smoothed_curves), run_time=4, rate_func=linear)
        self.wait(2)

        # Slowly rotate the camera to show off the 3D perspective
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4)
        self.stop_ambient_camera_rotation()

