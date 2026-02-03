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
        self.plotting_config.plot_context.__call__()
        self.__render_plots()
        self.plotting_config.plot_type.__call__()
        self.plotting_config.x_label_renderer.__call__()
        self.plotting_config.y_label_renderer.__call__()
        self.plotting_config.comment_renderer.__call__()
        self.plotting_config.annotate_renderer.__call__()
        plt.show()

    """ --------------------   Rendering of components -----------------------  """

    def __render_title(self) -> None:
        text, font_dict = self.plotting_config.title_renderer()
        plt.title(label=text, fontdict=font_dict)

    def __render_plots(self) -> None:
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

    def __render_grid(self) -> None:
        plt.grid(self.plotting_config.grid)

    def __render_legend(self) -> None:
        plt.legend(prop={'family': self.plotting_config.legend_renderer.font_family,
                         'size': self.plotting_config.legend_renderer.font_size,
                         'weight': self.plotting_config.legend_renderer.font_weight}
                   )

    def __render__annotations(self) -> None:
        plt.annotate(text=self.plotting_config.annotate_renderer.text,
                     xy=self.plotting_config.annotate_renderer.xy,
                     xytext=self.plotting_config.annotate_renderer.xytext,
                     color=self.plotting_config.annotate_renderer.color,
                     arrowprops=dict(arrowstyle=self.plotting_config.annotate_renderer.arrow_style,
                                     color='black',
                                     connectionstyle=self.plotting_config.annotate_renderer.connection_style))

    def __render_comments(self) -> None:
        if self.plotting_config.comment_renderer is not None:
            text, font_dict = self.plotting_config.comment_renderer()
            x, y = self.plotting_config.comment_renderer.position
            plt.text(x=x,
                     y=y,
                     s=text,
                     c=self.plotting_config.comment_renderer.font_color,
                     fontsize=self.plotting_config.comment_renderer.font_size,
                     fontweight=self.plotting_config.comment_renderer.font_weight,
                     transform=plt.gca().transAxes)

    def __render_labels(self) -> None:
        text, font_dict = self.plotting_config.x_label_renderer()
        plt.xlabel(xlabel=text, fontdict=font_dict)
        text, font_dict = self.plotting_config.y_label_renderer()
        plt.ylabel(ylabel=text, fontdict=font_dict)

    def __render_context(self) -> None:
        fig = plt.figure(figsize=self.plotting_config.plot_context.fig_size)
        fig.set_facecolor(self.plotting_config.plot_context.background_color)
        plt.grid(self.plotting_config.plot_context.grid)
        if self.plotting_config.plot_context.filename is not None:
            fig.savefig(f'{self.plotting_config.plot_context.filename}.png')





