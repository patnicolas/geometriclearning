__author__ = "Patrick R. Nicolas"
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

# Standard Library imports
from typing import Dict, AnyStr, Any
import logging
import python
# 3rd Party imports
import numpy as np
import matplotlib.pyplot as plt
from topology.homology.persistence_diagrams import PersistenceDiagrams
from topology.homology.shaped_data_generator import ShapedDataGenerator

__all__ = ['PersistentHomology']


class PersistentHomology(object):
    """
    Wrapper to evaluate the Persistent Homology implementation in scikit-tda - Topological Data Analysis library.
    scikit-tda leverage the Ripser library
    Documentation reference: https://docs.scikit-tda.org/en/latest/

    The data is synthetically generated from a shape (Torus, Sphere,) with additive noise.
    """
    num_shape_data_point = 96000
    size_raw_data_point = 140
    size_shaped_data_point = 36

    def __init__(self, shaped_data_generator: ShapedDataGenerator) -> None:
        """
        Constructor for the persistent homology evaluator.

        @param shaped_data_generator: Shape associated with the data to be used in the homology
        @type shaped_data_generator: ShapedDataGenerator
        """
        self.shaped_data_generator = shaped_data_generator

    def create_data(self, props: Dict[AnyStr, Any]) -> (AnyStr, np.array, np.array):
        """
        Create data using a dictionary descriptor for the shaped data generator. The generator is defined by the
        enumerator ShapedDataGenerator

        @param props: Configuration parameters for the shaped data generator
        @type props: Dict[AnyStr, Any]
        @return: Tuple (data shape type, shaped data, raw data with noise)
        @rtype: Tuple[AnyStr, np.array, np.array]
        """
        raw_data, shape_type = self.shaped_data_generator(props)
        props['noise'] = 0.0
        props['n'] = PersistentHomology.num_shape_data_point
        denoised_data, _ = self.shaped_data_generator(props)
        return shape_type, denoised_data, raw_data

    def plot(self, props: Dict[AnyStr, Any]) -> None:
        """
        Generate a 2D or 3D scatter plot with raw (noisy) data and shaped data using the enumerator ShapedDataGenerator
        @param props:
        @type props:
        """
        shape_type, shaped_data, raw_data = self.create_data(props)
        fig = plt.figure(figsize=(8, 8))

        match self.shaped_data_generator:
            # 2D scatter plot
            case ShapedDataGenerator.CIRCLE:
                PersistentHomology.__plot2d(shape_type, shaped_data, raw_data)
            # 3D scatter plot
            case ShapedDataGenerator.TORUS | ShapedDataGenerator.SWISS_ROLL | ShapedDataGenerator.SPHERE:
                PersistentHomology.__plot3d(shaped_data, raw_data, fig)

        plt.title(label=shape_type,  fontdict={'family': 'serif', 'size': 23, 'weight': 'bold', 'color': 'blue'})
        plt.show()

    def persistence_diagram(self, props: Dict[AnyStr, Any]) -> None:
        shape_type, data = self.shaped_data_generator(props)
        # Persistence diagrams are computation intensive so we need to limit the amount of data to be used.
        if len(data) >= 2048:
            logging.warning(f'Number of data points for persistence diagram {len(data)} truncated to 2048')
            data = data[:2048]

        # Instantiate the persistence diagram of given type
        persistence_diagram = PersistenceDiagrams(data, shape_type)
        # Display diagram
        persistence_diagram.display()

    """ ---------------------  Private supporting methods ------------------------- """

    @staticmethod
    def __plot2d(shape_type: AnyStr, shaped_data: np.array, raw_data: np.array) -> None:
        plt.scatter(x=shaped_data[:, 0],
                    y=shaped_data[:, 1],
                    label=f'{shape_type} shaped data',
                    s=PersistentHomology.size_shaped_data_point)
        plt.scatter(x=raw_data[:, 0],
                    y=raw_data[:, 1],
                    label=f'{shape_type} raw data',
                    s=PersistentHomology.size_raw_data_point)
        plt.legend()

    @staticmethod
    def __plot3d(shaped_data: np.array, raw_data: np.array, fig) -> None:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection='3d')
        fig.set_facecolor('#f0f9ff')
        ax.set_facecolor('#f0f9ff')
        ax.scatter(xs=shaped_data[:, 0],
                   ys=shaped_data[:, 1],
                   zs=shaped_data[:, 2],
                   color='grey',
                   alpha=0.06,
                   s=PersistentHomology.size_shaped_data_point,
                   edgecolor='none')
        ax.scatter(xs=raw_data[:, 0],
                   ys=raw_data[:, 1],
                   zs=raw_data[:, 2],
                   color='red',
                   alpha=1.0,
                   s=PersistentHomology.size_raw_data_point,
                   edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

