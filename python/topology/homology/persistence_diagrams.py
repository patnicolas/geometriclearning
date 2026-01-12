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
from typing import AnyStr, Dict, Any, Self
# 3rd Party imports
import numpy as np
import matplotlib.pyplot as plt
# Library imports
from topology.homology.shaped_data_generator import ShapedDataGenerator
from topology.homology.persistence_diagram_type import BirthDeathDiagram, PersistenceLandscape, PersistenceImage

class PersistenceDiagrams(object):
    """
    Wrapper to create and visualize any persistence diagram.
    The persistence diagrams are implemented in the class inherited from the PersistenceDiagramType
    ref: homology.persistence_diagram_type

    The 4 persistence diagrams currently supported are
    - Birth & death diagram - class BirthDeathDiagram
    - Persistence image - class PersistenceImage
    - Exact persistence landscape - class PersistenceLandscape
    - Approximation persistence landscape
    """
    num_shape_data_point = 48000

    def __init__(self, data: np.array, data_shape: AnyStr = None) -> None:
        """
        Default constructor for the generation and visualization of persistence diagrams.
        :param data: Input data to be represented
        :type data: Numpy array
        :param data_shape: Descriptor for the shape of the data
        :type data_shape: AnyStr
        """
        self.data = data
        self.data_shape = data_shape

    @classmethod
    def build(cls, props: Dict[AnyStr, Any], shaped_data_generator: ShapedDataGenerator) -> Self:
        """
        Create data using a dictionary descriptor for the shaped data generator. The generator is defined by the
        enumerator ShapedDataGenerator

        @param props: Configuration parameters for the shaped data generator
        @type props: Dict[AnyStr, Any]
        @param shaped_data_generator: Shape associated with the data to be used in the homology
        @type shaped_data_generator: ShapedDataGenerator
        @return: Instance of Persistence Diagrams
        @rtype: PersistenceDiagrams
        """
        shaped_data, shape_type = shaped_data_generator(props)
        return cls(shaped_data, shape_type)

    def display(self) -> None:
        """
        Method to display the various persistence diagrams into a single 2 x 2 matplotlib plot. The steps are
        1 Set up the plotting environment
        2 Instantiation of Vietoris-Rips complex
        3 Subplot Persistence image
        4 Subplot Approximate persistence landscape
        5 Subplot Exact persistence landscape
        6 Subplot Birth - death
        """
        from ripser import Rips

        # Global plotting configuration
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        fig.set_facecolor('#f0f9ff')
        data_shape = self.data_shape if self.data_shape is not None else 'Unidentified'
        fig.suptitle(f'Persistence Diagrams:  {data_shape}', fontsize=15, color='red')

        # Instantiate the Vietoris-Rips complex
        rips = Rips(maxdim=2)

        # Plot Persistence Image
        persistent_image = PersistenceImage(axes[0][1], self.data)
        persistent_image.display(title='Persistence Image', x_label='Birth', y_label='Persistence')

        # Approximate Landscape
        approx_landscape_diagram = PersistenceLandscape(ax=axes[1][0],
                                                        diagram_data=self.data,
                                                        hom_degree=0,
                                                        depth=6,
                                                        is_exact=False)
        approx_landscape_diagram.display(title='Approximation Landscape ',
                                         x_label='Filtration parameter degree ',
                                         y_label='Function Value')

        # Exact Landscape
        exact_landscape_diagram = PersistenceLandscape(ax=axes[1][1],
                                                       diagram_data=self.data,
                                                       hom_degree=1,
                                                       depth=6,
                                                       is_exact=True)
        exact_landscape_diagram.display(title='True Landscape ',
                                        x_label='Filtration parameter degree ',
                                        y_label='Function Value')

        # Persistence diagram (Birth - Death)
        birth_death_diagram = BirthDeathDiagram(axes[0][0], rips.transform(self.data))
        birth_death_diagram.display(title='Persistent Diagram', x_label='Birth', y_label='Death')

    """ --------------------- Private supporting methods ------------------- """

    @staticmethod
    def __set_plot_env(ax, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        ax.set_xlabel(x_label, fontdict={'family': 'serif', 'size': 11,  'style': 'italic'})
        ax.set_ylabel(y_label, fontdict={'family': 'serif', 'size': 11,  'style': 'italic'})
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(label=title, fontdict={'family': 'serif', 'size': 14, 'color': 'blue'})
