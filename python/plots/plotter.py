__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from typing import List, AnyStr, NoReturn, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch

"""
    Wraps the parameters for plots. The static methods generated a '.png' file which name is time stamped.
"""


@dataclass
class PlotterParameters:
    count: int
    x_label: AnyStr
    y_label: AnyStr
    title: AnyStr
    fig_size: Optional[Tuple[int, int]] = None
    time_str = datetime.now().strftime("%b-%d-%Y:%H.%M")
    """
        Constructor
        @param count: Count such as number of epoch, iteration or step
        @param x_label: Label for X-axis
        @param y_label: Label for Y_axis
        @param title: Title for the plot
    """

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Plotter(object):
    images_folder = '../../../../tests/images/'
    markers = ['-', '--', '-.', '+', 'bD--', '^']
    colors = ['blue', 'green', 'red', 'black', 'orange', 'grey']

    @staticmethod
    def single_plot_np_array(np_array1: np.array, np_array2: np.array, plotter_parameters: PlotterParameters):
        fig, axes = plt.subplots()
        axes.plot(np_array1, np_array2)
        axes.set(xlabel=plotter_parameters.x_label, ylabel=plotter_parameters.y_label, title=plotter_parameters.title)
        axes.grid()
        fig.savefig(f"{Plotter.images_folder}/plot-{plotter_parameters.time_str}.png")
        plt.show()

    @staticmethod
    def single_plot(values1: List[float], plotter_parameters: PlotterParameters) -> NoReturn:
        fig, axes = plt.subplots()
        x = np.arange(0, len(values1), 1)
        y = np.asarray(values1)
        axes.plot(x, y)
        axes.set(xlabel=plotter_parameters.x_label, ylabel=plotter_parameters.y_label, title=plotter_parameters.title)
        axes.grid()
        fig.savefig(f"{Plotter.images_folder}/plot-{plotter_parameters.time_str}.png")
        plt.show()

    @staticmethod
    def multi_plot(
            dict_values: Dict[AnyStr, List[torch.Tensor]],
            plotter_params_list: List[PlotterParameters]) -> NoReturn:
        """
         Generic 1, 2, or 3 sub-plots with one variable value
         @param dict_values: Dictionary of array of floating values
         @type dict_values:  Dict[AnyStr, List[tensor]]
         @param plotter_params_list: List of plotting parameters
         @type plotter_params_list: List[PlotterParameters]
         """
        num_points = Plotter.__validate_params(dict_values, plotter_params_list)
        fig, axes = plt.subplots(ncols=1, nrows=len(dict_values), figsize=plotter_params_list[0].fig_size)
        x = np.arange(0, num_points, 1)
        for plot_index in range(len(dict_values)):
            title = plotter_params_list[plot_index].title
            y = dict_values[title]
            Plotter.__axis_plot(x, plotter_params_list[plot_index], y, axes, plot_index)

        fig.savefig(f"{Plotter.images_folder}/plot-{Plotter.time_str()}.png")
        plt.show()

    @staticmethod
    def plot(values: List[List[float]], labels: List[AnyStr], plotter_parameters: PlotterParameters) -> NoReturn:
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
        plt.title(plotter_parameters.title)
        plt.xlabel(plotter_parameters.x_label)
        plt.ylabel(plotter_parameters.y_label)
        plt.legend()
        plt.show()

    @staticmethod
    def time_str() -> str:
        return datetime.now().strftime("%b-%d-%Y-%H.%M.%S")

    """ ------------ Helper private methods --------------- """

    @staticmethod
    def __two_plot(values1: List[float], values2: List[float], plotter_parameters_list: List[PlotterParameters]) -> NoReturn:
        assert len(plotter_parameters_list) == 2, f'Number of plots {plotter_parameters_list.count} should be 2'
        assert len(values1) == len(values2), f'Number of values1 {len(values1)} != number values2 {len(values2)}'

        fig, axes = plt.subplots(2)
        x = np.arange(0, len(values1), 1)
        for i in range(2):
            Plotter.__axis_plot(x, plotter_parameters_list[i], values1, axes, i)
        fig.savefig(f"{Plotter.images_folder}/plot-{Plotter.time_str()}.png")
        plt.show()

    @staticmethod
    def __three_plot(
            values1: List[float],
            values2: List[float],
            values3: List[float],
            plotter_parameters_list: List[PlotterParameters]) -> NoReturn:
        assert len(plotter_parameters_list) == 3, f'Number of plots {plotter_parameters_list.count} should be 3'
        assert len(values1) == len(values2), f'Size of features {len(values1)} should be == Size labels {len(values2)}'
        assert len(values1) == len(values3), f'Size of features {len(values1)} should be == Size z {len(values3)}'

        fig, axes = plt.subplots(3)
        x = np.arange(0, len(values1), 1)
        for i in range(3):
            Plotter.__axis_plot(x, plotter_parameters_list[i], torch.Tensor(values1), axes, i)
        fig.savefig(f"{Plotter.images_folder}/plot-{Plotter.time_str()}.png")
        plt.show()

    @staticmethod
    def __axis_plot(
            x: np.array,
            plotter_param: PlotterParameters,
            torch_values: List[torch.Tensor],
            axes: np.array,
            index: int) -> NoReturn:
        values = [value.cpu().float() for value in torch_values]
        y = np.asarray(values)
        axes[index].plot(x, y)
        axes[index].set(xlabel=plotter_param.x_label, ylabel=plotter_param.y_label, title=plotter_param.title)
        axes[index].grid()


    @staticmethod
    def __validate_params(
            dict_values: Dict[AnyStr, List[torch.Tensor]],
            plotter_params_list: List[PlotterParameters]) -> int:
        num_plots = len(dict_values)
        assert len(plotter_params_list) == num_plots, f'Number of plots {len(plotter_params_list)} should be {num_plots}'
        v_0 = None
        for k, v in dict_values.items():
            if v_0 is None:
                v_0 = v
            else:
                assert len(v) == len(v_0),  f'Num. values for {k}: {len(v)} should be {len(v_0)}'
        return len(v_0)


