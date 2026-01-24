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

from typing import AnyStr, Dict
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from plots.plotting_engine import PlottingEngine
from plots.plotting_config import PlottingConfig


class MatplotlibEngine(PlottingEngine):
    markers = ['-', '--', '-.', '--', ':', '-']
    colors = ['blue', 'green', 'red', 'black', 'grey', 'orange']

    def __init__(self, data_dict: Dict[AnyStr, np.array], plotting_config: PlottingConfig | AnyStr) -> None:
        super(MatplotlibEngine, self).__init__(data_dict, plotting_config)

    def __str__(self) -> AnyStr:
        return f'{str(self.data_dict.items())}\n{self.plotting_config}'

    def render(self) -> None:
        match self.plotting_config.plot_type:
            case 'line_plots':
                self.__render_line_plots()
            case _:
                raise NotImplementedError(f'{self.plotting_config.plot_type} not implemented')

    def save(self, filename: AnyStr, fig: Figure) -> None:
        if self.plotting_config.filename is not None:
            fig.savefig(f'{self.plotting_config.filename}.png')

    def __render_line_plots(self) -> None:
        fig = plt.figure(self.plotting_config.fig_size)
        fig.set_facecolor(self.plotting_config.background_color)

        iterator = iter(self.data_dict.items())
        _, x_values = next(iterator)

        # Draws the multiple plots
        count = 0
        for y_label, y_value in iterator:
            plt.plot(x_values,
                     y_value,
                     label=y_label,
                     color=MatplotlibEngine.colors[count],
                     linestyle=MatplotlibEngine.markers[count])
            count += 1
        text, font_dict = self.plotting_config.title_config()
        plt.title(label=text, fontdict=font_dict)
        text, font_dict = self.plotting_config.x_label_config()
        plt.xlabel(xlabel=text, fontdict=font_dict)
        text, font_dict = self.plotting_config.y_label_config()
        plt.ylabel(ylabel=text, fontdict=font_dict)

        if self.plotting_config.comment_config is not None:
            text, font_dict = self.plotting_config.comment_config()
            x, y = self.plotting_config.comment_config.position
            plt.text(x=x,
                     y=y,
                     s=text,
                     c=self.plotting_config.comment_config.font_color,
                     fontsize=self.plotting_config.comment_config.font_size,
                     fontweight=self.plotting_config.comment_config.font_weight,
                     transform=plt.gca().transAxes)

        plt.legend(prop={'family': self.plotting_config.x_label_config.font_type,
                         'size': self.plotting_config.get_legend_font_size()})
        # Draw a grid only if defined in the specified in configuration
        plt.grid(self.plotting_config.grid)
        plt.show()






