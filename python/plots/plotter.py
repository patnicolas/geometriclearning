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

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from typing import List, AnyStr, Tuple, Optional, Dict, Any, Self
from dataclasses import dataclass
import torch
__all__ = ['PlotterParameters', 'Plotter']


@dataclass
class PlotterParameters:
    """
        Wraps the parameters for plots. The static methods generated a '.png' file which name is time stamped.
    """
    count: int
    x_label: AnyStr
    y_label: AnyStr
    title: AnyStr
    is_image: Optional[bool] = False
    fig_size: Optional[Tuple[int, int]] = None
    time_str = datetime.now().strftime("%b-%d-%Y:%H.%M")
    """
    Constructor
    @param count: Count such as number of epoch, iteration or step
    @type count: int
    @param x_label: Label for X-axis
    @type x_label: str
    @param y_label: Label for Y_axis
    @type y_label: str
    @param title: Title for the plot
    @type title: str
    @param is_image: Specify if this is an image to be displayed optional
    @type is_image: bool
    @param fig_size: Optional figure size
    @type fig_size: (int, int)
    @param time_str: Time stamp the plot was created, used for name of the file the plot image is stored
    @type time_str: str
    """

    def __repr__(self) -> AnyStr:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> AnyStr:
        return f'   Title:   {self.title}\n   X label: {self.x_label}\n   Y label: {self.y_label}'

    @classmethod
    def build(cls, attributes: Dict[AnyStr, Any]) -> Self:
        assert attributes.__contains__('count'), 'count is undefined as plot attributes'
        
        count = attributes['count']
        x_label = attributes.get('x_label', 'X')
        y_label = attributes.get('y_label', 'Y')
        title = attributes.get('title', '')
        return cls(count, x_label, y_label, title)


class Plotter(object):
    plot_parameters_label = 'plot_parameters'
    test_images_folder = '../../output_images'
    markers = ['-', '--', '-.', '--', '^', '-']
    colors = ['blue', 'green', 'red', 'orange', 'black', 'grey']

    @staticmethod
    def single_plot_np_array(np_array1: np.array, np_array2: np.array, plotter_parameters: PlotterParameters):
        fig, axes = plt.subplots()
        axes.plot(np_array1, np_array2)
        axes.set(xlabel=plotter_parameters.x_label, ylabel=plotter_parameters.y_label, title=plotter_parameters.title)
        axes.grid()
        fig.savefig(f"{Plotter.test_images_folder}/plot-{plotter_parameters.time_str}.png")
        plt.show()

    @staticmethod
    def single_plot(values1: List[float], plotter_parameters: PlotterParameters) -> None:
        fig, axes = plt.subplots()
        x = np.arange(0, len(values1), 1)
        y = np.asarray(values1)
        axes.plot(x, y)
        axes.set(xlabel=plotter_parameters.x_label, ylabel=plotter_parameters.y_label, title=plotter_parameters.title)
        axes.grid()
        fig.savefig(f"{Plotter.test_images_folder}/plot-{plotter_parameters.time_str}.png")
        plt.show()

    @staticmethod
    def multi_plot(dict_values: Dict[AnyStr, List[float]],
                   plotter_params_list: List[PlotterParameters],
                   plot_title: AnyStr) -> None:
        """
        Generic 1, 2, or 3 sub-plots with one variable value
        @param dict_values: Dictionary of array of floating values
        @type dict_values:  Dict[AnyStr, List[tensor]]
        @param plotter_params_list: List of plotting parameters
        @type plotter_params_list: List[PlotterParameters]
        @param plot_title: Title for plot
        @type plot_title: str
        """
        num_points = Plotter.__validate_params(dict_values, plotter_params_list)
        num_dict_values = len(dict_values)
        num_rows = num_dict_values // 2 if num_dict_values % 2 == 0 else (num_dict_values // 2) + 1
        fig, axes = plt.subplots(ncols=2, nrows=num_rows, figsize=plotter_params_list[0].fig_size)
        x = np.arange(0, num_points, 1)

        plot_index = 0
        for col_index in range(2):
            for row_index in range(num_rows):
                if plot_index < num_dict_values:
                    key = plotter_params_list[plot_index].y_label
                    y = dict_values[key]
                    x_lim = len(y)
                    y_max = max(y)
                    y_lim = 1.0 if y_max <= 1.0 else y_max
                    Plotter.__multi_axis_plot(x,
                                              plotter_params_list[plot_index],
                                              y,
                                              axes,
                                              (row_index, col_index),
                                              (x_lim, y_lim))
                plot_index += 1

        font_style = {'family': 'sans-serif', 'size': 16}
        fig.suptitle(plot_title, **font_style)
        plt.tight_layout()
        # plt.show()
        plot_file_name = f"{Plotter.test_images_folder}/plot_{plot_title}.png"
        fig.savefig(plot_file_name)

    @staticmethod
    def plot(values: List[List[float]], labels: List[AnyStr], plotter_parameters: PlotterParameters) -> None:
        """
        Display values from multiple variables  on a single plot
        @param values: List of array of values
        @type values: List[List[float]]
        @param labels: List of labels, one per variable
        @type labels: List[str]
        @param plotter_parameters: Plotter parameters (title, legend,..)
        @type plotter_parameters: PlotterParameters
        """
        assert len(values) == len(labels), f'Number of variables {len(values)} != number of labels {len(labels)}'

        len_x = len(values[0])
        x = np.arange(0, len_x, 1)
        y = [np.array(vals) for vals in values]
        fig_size = plotter_parameters.fig_size if plotter_parameters.fig_size is not None else (12, 12)
        plt.figure(figsize=fig_size)

        for i in range(len(y)):
            plt.plot(x, y[i], label=labels[i], color=Plotter.colors[i], linestyle=Plotter.markers[i])

        plt.title(
            plotter_parameters.title,
            fontdict={'family': 'sans-serif', 'size': 22, 'weight': 'bold'}
        )
        plt.xlabel(
            plotter_parameters.x_label,
            fontdict={'family': 'serif', 'size': 16, 'style': 'italic'}
        )
        plt.ylabel(
            plotter_parameters.y_label,
            fontdict={'family': 'serif', 'size': 16, 'style': 'italic'}
        )
        plt.legend(prop={'family': 'serif', 'size': 14})
        plt.show()

    @staticmethod
    def time_str() -> str:
        return datetime.now().strftime("%b-%d-%Y-%H.%M.%S")

    """ ------------ Helper private methods --------------- """

    @staticmethod
    def __two_plot(values1: List[float],
                   values2: List[float],
                   plotter_parameters_list: List[PlotterParameters]) -> None:
        assert len(plotter_parameters_list) == 2, f'Number of plots {plotter_parameters_list.count} should be 2'
        assert len(values1) == len(values2), f'Number of values1 {len(values1)} != number values2 {len(values2)}'

        fig, axes = plt.subplots(2)
        x = np.arange(0, len(values1), 1)
        for i in range(2):
            Plotter.__axis_plot(x, plotter_parameters_list[i], [torch.Tensor(x) for x in values1], axes, i)
        fig.savefig(f"{Plotter.test_images_folder}/plot-{Plotter.time_str()}.png")
        plt.show()

    @staticmethod
    def __three_plot(
            values1: List[float],
            values2: List[float],
            values3: List[float],
            plotter_parameters_list: List[PlotterParameters]) -> None:
        assert len(plotter_parameters_list) == 3, f'Number of plots {plotter_parameters_list.count} should be 3'
        assert len(values1) == len(values2), f'Size of features {len(values1)} should be == Size labels {len(values2)}'
        assert len(values1) == len(values3), f'Size of features {len(values1)} should be == Size z {len(values3)}'

        fig, axes = plt.subplots(3)
        x = np.arange(0, len(values1), 1)
        for i in range(3):
            Plotter.__axis_plot(x, plotter_parameters_list[i], [torch.Tensor(x) for x in values1], axes, i)
        # fig.savefig(f"{Plotter.test_images_folder}/plot-{Plotter.time_str()}.png")
        plt.show()

    @staticmethod
    def __multi_axis_plot(
            x: np.array,
            plotter_param: PlotterParameters,
            values: List[float],
            axes,
            index: (int, int),
            limits: (int, int)) -> None:
        y = np.asarray(values)
        limit_y, delta_x, delta_y = Plotter.arrange_y(limits)
        axes[index[0]][index[1]].set_facecolor('black')
        axes[index[0]][index[1]].set_xlim(1, limits[0]-1)
        axes[index[0]][index[1]].set_ylim(0, limit_y)
        axes[index[0]][index[1]].set_xticks(np.arange(0, limits[0], delta_x))
        axes[index[0]][index[1]].set_yticks(np.arange(0,  limit_y, delta_y))
        axes[index[0]][index[1]].plot(x, y, color='yellow')
        axes[index[0]][index[1]].set(xlabel=plotter_param.x_label, ylabel=plotter_param.y_label, title='')
        axes[index[0]][index[1]].xaxis.label.set_fontsize(10)
        axes[index[0]][index[1]].tick_params(axis='x', labelsize=10, labelrotation=0)
        axes[index[0]][index[1]].yaxis.label.set_fontsize(12)
        axes[index[0]][index[1]].yaxis.label.set_fontweight('bold')
        axes[index[0]][index[1]].yaxis.label.set_color('black')
        axes[index[0]][index[1]].tick_params(axis='y', labelsize=10)
        axes[index[0]][index[1]].grid(which='major', color='lightgray', linestyle='-', linewidth=0.7)

    @staticmethod
    def arrange_y(limits: (int, int)) -> (float, float, float):
        y_int_max = int(limits[1]-0.01)+1
        delta_y = 0.2*y_int_max
        delta_x = int(limits[0]*0.1+1)
        return y_int_max,  delta_x, delta_y

    @staticmethod
    def __axis_plot(
            x: np.array,
            plotter_param: PlotterParameters,
            values: List[float],
            axes,
            index: int,
            last_plot: bool) -> None:
        y = np.asarray(values)
        axes[index].plot(x, y)
        axes[index].set(xlabel=plotter_param.x_label, ylabel=plotter_param.y_label, title='')
        if last_plot:
            axes[index].xaxis.label.set_fontsize(10)
            axes[index].tick_params(axis='x', labelsize=12, labelrotation=0)
        else:
            axes[index].xaxis.label.set_fontsize(10)
            axes[index].tick_params(axis='x', labelsize=1, labelrotation=0)
        axes[index].yaxis.label.set_fontsize(12)
        axes[index].tick_params(axis='y', labelsize=12)
        axes[index].grid()

    @staticmethod
    def __validate_params(
            dict_values: Dict[AnyStr, List[float]],
            plotter_params_list: List[PlotterParameters]) -> int:
        from metric.performance_metrics import PerformanceMetrics

        num_plots = len(dict_values)
        assert len(plotter_params_list) == num_plots, f'Number of plots {len(plotter_params_list)} should be {num_plots}'
        v_0 = None
        for k, v in dict_values.items():
            if k in PerformanceMetrics.valid_metrics and v_0 is None:
                v_0 = v
        return len(v_0)



