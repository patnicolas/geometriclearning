__author__ = "Patrick R. Nicolas"
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
from typing import AnyStr
# 3rd Party imports
import persim
import numpy as np
import matplotlib.pyplot as plt



class PersistenceDiagram(object):

    def __init__(self, data: np.array, data_shape: AnyStr = None) -> None:
        self.data = data
        self.data_shape = data_shape

    def display(self) -> None:
        import ripser
        from ripser import Rips
        from persim import PersistenceImager, PersLandscapeApprox, PersLandscapeExact
        from persim.landscapes import plot_landscape_simple

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        data_shape = self.data_shape if self.data_shape is not None else 'Unidentified'
        fig.suptitle(f'Persistence Diagrams:  {data_shape}', fontsize=15, color='red')

        # Instantiate the rips complex
        rips = Rips()
        rips_data = rips.transform(self.data)

        # Second plot: Persistence Image
        dgms = ripser.ripser(self.data)['dgms']
        pimager = PersistenceImager(pixel_size=0.3)
        input_data = dgms[1:3]
        pimager.fit(input_data)
        output_image = pimager.transform(input_data)
        axes[0][1].matshow(output_image[0].T, **{"origin": "lower"})
        self.__set_plot_env(axes[0][1], title='Persistence Image', x_label='Birth', y_label='Persistence')

        # 3rd plot: Approximate Landscape
        self.__set_plot_env(ax=axes[1][0],
                            title='Approximation Landscape ',
                            x_label='Filtration parameter degree 0',
                            y_label='Function Value')
        pla = PersLandscapeApprox(dgms=dgms, hom_deg=0)
        plot_landscape_simple(pla, ax=axes[1][0], depth_range=range(6))

        # 4th plot: Exact Landscape
        self.__set_plot_env(ax=axes[1][1],
                            title='Exact Landscape ',
                            x_label='Filtration parameter degree 1',
                            y_label='Function Value')
        pla = PersLandscapeExact(dgms=rips_data, hom_deg=1)
        plot_landscape_simple(pla, depth_range=range(6))

        # First plot: Persistence diagram (Birth - Death)
        self.__set_plot_env(axes[0][0], title='Persistent Diagram', x_label='Birth', y_label='Death')
        persim.plot_diagrams(show=True, diagrams=rips_data, ax=axes[0][0])

    """ --------------------- Private supporting methods ------------------- """

    @staticmethod
    def __set_plot_env(ax, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        ax.set_xlabel(x_label, fontdict={'family': 'serif', 'size': 11,  'style': 'italic'})
        ax.set_ylabel(y_label, fontdict={'family': 'serif', 'size': 11,  'style': 'italic'})
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(label=title, fontdict={'family': 'serif', 'size': 14, 'color': 'blue'})

