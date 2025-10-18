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


# Standard Library imports
from typing import List, AnyStr, Tuple, Optional, Dict, Any, Self
from dataclasses import dataclass
import logging
# 3rd Party imports
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import python
__all__ = ['MetricPlotterParameters', 'MetricPlotter']


@dataclass(frozen=True)
class MetricPlotterParameters:
    """
    Wraps the parameters for plots. The static methods generated a '.png' file which name is time stamped.

    @param count: Count such as number of epoch, iteration or step
    @type count: int
    @param x_label: Label for X-axis
    @type x_label: str
    @param x_label_size: Size of font for x-axis label
    @type x_label_size: int
    @param title: Title for the plot
    @type title: str
    @param fig_size: Optional figure size
    @type fig_size: (int, int)
    @param time_str: Time stamp the plot was created, used for name of the file the plot image is stored
    @type time_str: str
    """
    count: int
    x_label: AnyStr
    title: AnyStr
    x_label_size: int
    plot_filename: AnyStr = None
    fig_size: Optional[Tuple[int, int]] = None
    time_str = datetime.now().strftime("%b-%d-%Y:%H.%M")

    @classmethod
    def build(cls, attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor using a dictionary
        @param attributes: Dictionary of Metric Plotter configuration parameters
        @type attributes: Dictionary
        @return: Instance of Metric Plotter Parameters
        @rtype: MetricPlotterParameters
        """
        return cls(count=attributes.get('count', 0),
                   x_label=attributes.get('x_label', 'X'),
                   title=attributes['title'],
                   x_label_size=attributes.get('x_label_size', 12),
                   plot_filename=attributes.get('plot_filename', 'default_metrics'),
                   fig_size=attributes.get('fig_size', (10, 8)))

    def save_plot(self, fig) -> bool:
        """
        Save the plot into file is plot_folder is defined

        @param fig: Current figure
        @return: True if plot to be saved in file, False if plot to be displayed
        """
        if self.plot_filename is not None:
            fig.savefig(f'{self.plot_filename}.png')
        return self.plot_filename is not None

    def __repr__(self) -> AnyStr:
        return (f'\nTitle: {self.title}\nX label: {self.x_label}\nX label Size: {self.x_label_size}'
                f'\nPlot Filename: {self.plot_filename}\nFig size: {self.fig_size}')


class MetricPlotter(object):
    plot_parameters_label = 'plot_parameters'
    # images_folder = '../../output_plots'
    markers = ['-', '--', '-.', '--', '^', '-']
    colors = ['blue', 'green', 'red', 'orange', 'black', 'grey']

    def __init__(self, plotter_params: MetricPlotterParameters) -> None:
        """
        Constructor for the Metric plot
        @param plotter_params: List of plotting parameters
        @type plotter_params: List[PlotterParameters]
        """
        self.plotter_params = plotter_params

    def plot(self, dict_values: Dict[AnyStr, List[float]]) -> None:
        """
        Generic 1, 2, or 3 sub-plots with one variable value
        @param dict_values: Dictionary of array of floating values
        @type dict_values:  Dict[AnyStr, List[tensor]]
        """
        num_dict_values = len(dict_values)
        assert num_dict_values > 0, 'Dictionary of values to be plotted is undefined'

        num_rows = num_dict_values // 2 if num_dict_values % 2 == 0 else (num_dict_values // 2) + 1
        fig, axes = plt.subplots(ncols=2, nrows=num_rows, figsize=self.plotter_params.fig_size)

        plot_index = 0
        for key, values in dict_values.items():
            row_index = plot_index // 2
            col_index = plot_index % 2
            self.__multi_axis_plot(key,
                                   np.arange(0, len(values), 1),
                                   values,
                                   axes,
                                    (row_index, col_index),
                                    (0, len(values)))
            plot_index += 1

        font_style = {'family': 'sans-serif', 'size': 16}
        fig.suptitle(self.plotter_params.title, **font_style)
        plt.tight_layout()

        # If we need to store the plot image....
        if not self.plotter_params.save_plot(fig):
            plt.show()

    def __multi_axis_plot(
            self,
            key: AnyStr,
            x: np.array,
            values: List[float],
            axes,
            index: (int, int),
            x_limits: (int, int)) -> None:

        y_low, y_high, delta_y = MetricPlotter.plots_bounds(values)
        y = np.asarray(values)
        delta_x = int((x_limits[1] - x_limits[0]) * 0.1) + 1

        color = '#030580' if key in ('TrainLoss', 'EvalLoss') else 'black'

        axes[index[0]][index[1]].set_facecolor(color)
        max_x = 2 if x_limits[0] <= 1 else x_limits[0] - 1
        axes[index[0]][index[1]].set_xlim(1, max_x)
        axes[index[0]][index[1]].set_ylim(y_low, y_high)
        axes[index[0]][index[1]].set_xticks(np.arange(x_limits[0], x_limits[1], delta_x))
        axes[index[0]][index[1]].set_yticks(np.arange(y_low, y_high, delta_y))
        axes[index[0]][index[1]].plot(x, y, color='yellow')
        axes[index[0]][index[1]].set(xlabel=self.plotter_params.x_label,
                                     ylabel=key,
                                     title='')
        axes[index[0]][index[1]].xaxis.label.set_fontsize(self.plotter_params.x_label_size)
        axes[index[0]][index[1]].tick_params(axis='x', labelsize=self.plotter_params.x_label_size, labelrotation=0)
        axes[index[0]][index[1]].yaxis.label.set_fontsize(12)
        axes[index[0]][index[1]].yaxis.label.set_fontweight('bold')
        axes[index[0]][index[1]].yaxis.label.set_color(color)
        axes[index[0]][index[1]].tick_params(axis='y', labelsize=self.plotter_params.x_label_size)
        axes[index[0]][index[1]].grid(which='major', color='lightgray', linestyle='-', linewidth=0.7)

    @staticmethod
    def plots_bounds(values: List[float]) -> (float, float, float):
        import math
        if len(values) > 0:
            floor_value = math.floor(min(values) * 10.0) * 0.1
            ceil_value = math.ceil(max(values) * 10.0) * 0.1
            delta = math.ceil(ceil_value - floor_value)
            delta_y = delta*0.1 if ceil_value > 1.0 else 0.1

            return round(floor_value, 1), round(ceil_value, 1), round(delta_y, 1)
        else:
            logging.warning('Values for computing bounds of plots are undefined')
            return 0.0, 1.0, 0.1
