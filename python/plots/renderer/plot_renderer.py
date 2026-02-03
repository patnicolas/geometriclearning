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

from typing import AnyStr, Optional, Tuple, Self, Any, Dict
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt

from plots.renderer import Renderer
from plots.renderer.component_renderer import (PlotsRenderer, PlotContext, LegendRenderer, CommentRenderer,
                                               AnnotationRenderer, TextRenderer)


class PlotRenderer(Renderer):
    """
    Generic data class for configuration any plot, independently of the plotting library or engine. The plot is
    automatically saved if the variable, filename is defined.

    @param plot_type: Type of the plot (scatter, line plot, bar chart ...).
    @type plot_type: str
    @param plot_context: Define the context for the plot (background color, figure size ...)
    @type plot_context: PlotContext
    @param line_plots_renderer: Renderer for the line plots
    @type line_plots_renderer: PlotsRenderer
    @param title_renderer: Configuration for the title of the plot
    @type title_renderer: TextRenderer
    @param x_label_renderer: Configuration for the X-label of the plot
    @type x_label_renderer: TextRenderer
    @param y_label_renderer: Configuration for the y-label of the plot
    @type y_label_renderer: CommentRenderer
    @param comment_renderer: Configuration for comment the plot
    @type comment_renderer: CommentRenderer
    """
    plot_type: AnyStr
    plot_context: PlotContext
    line_plots_renderer: PlotsRenderer
    title_renderer: TextRenderer
    x_label_renderer: TextRenderer
    y_label_renderer: TextRenderer
    legend_renderer: LegendRenderer
    comment_renderer: CommentRenderer = None
    annotate_renderer: AnnotationRenderer = None

    @classmethod
    def build(cls, json_config_str: AnyStr) -> Self:
        """
        Alternative constructor taking a JSON string as input to deserialize
        Raise a JSONDecodeError exception if case the JSON string is malformed

        @param json_config_str: Input JSON string
        @type json_config_str: str
        @return: Instance of the plotting tconfiguration
        @rtype: PlotRenderer
        """
        config_dict = json.loads(json_config_str)
        return PlotRenderer(**config_dict)

    def to_json(self) -> AnyStr:
        return json.dumps(asdict(self), indent=2)

    def draw(self, arg: Any) -> None:
        self.plot_context()
        self.line_plots_renderer()
        self.title_renderer('title')
        self.x_label_renderer('xlabel')
        self.y_label_renderer('ylabel')
        if self.comment_renderer is not None:
            self.comment_renderer()
        if self.annotate_renderer is not None:
            self.annotate_renderer()
        plt.show()

    def __str__(self) -> AnyStr:
        return (f'\nType: {self.plot_type}\nTitle: {self.title_renderer}\nX-label: {self.x_label_renderer}'
                f'\nY-Label: {self.y_label_renderer}\nLegend: {self.legend_renderer}\nContext: {self.plot_context}'
                f'\nComment: {self.comment_renderer}')
