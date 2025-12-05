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
from typing import Dict, AnyStr, Any
from enum import Enum, unique
import logging
import python
# 3rd Party imports
import tadasets
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['ShapedDataGenerator', 'PersistentHomology']

from topology.homology.persistence_diagrams import PersistenceDiagram


@unique
class ShapedDataGenerator(Enum):
    """
    Enumerator for generation of shaped data with default values.
    The lambdas take a dictionary as input and output a tuple (Shape type, data generator)
        { } -> (Shape_type, shaped data generator)

    Example of input dictionary:
        { 'n': 250, 'noise': 0.4, 'c': 8}
    """
    CIRCLE = lambda k: (f"Circle {ShapedDataGenerator.__noise_label(k)}",
                        tadasets.dsphere(d=1, n=k.get('n', 100), noise=k.get('noise', 0.0)))
    SPHERE = lambda k: (f"Sphere {ShapedDataGenerator.__noise_label(k)}",
                        tadasets.sphere(n=k.get('n', 100), noise=k.get('noise', 0.0)))
    TORUS = lambda k: (f"Torus {ShapedDataGenerator.__noise_label(k)}",
                       tadasets.torus(n=k.get('n', 100),
                                      c=k.get('c', 10),
                                      a=k.get('a', 0.2),
                                      noise=k.get('noise', 0.0)))
    SWISS_ROLL = lambda k: (f"Swiss Roll {ShapedDataGenerator.__noise_label(k)}",
                            tadasets.swiss_roll(n=k.get('n', 100), noise=k.get('noise', 0.0)))

    def __call__(self, *args, **kwargs) -> (AnyStr, np.array):
        """
        Method to return the lambda associated to a shape. The parameter values are validated prior execution of
        lambda.

        @param args: Arguments list
        @type args: List[Any]
        @param kwargs: Arguments dictionary
        @type kwargs: Dict[AnyStr, Any]
        @return: Tuple (Shape type, shaped data)
        @rtype: Tuple
        """
        ShapedDataGenerator.__validate(kwargs)
        return self.value(*args, **kwargs)

    @staticmethod
    def __noise_label(k: Dict[AnyStr, Any]) -> AnyStr:
        noise = k.get('noise', 0.0)
        return f"with {k.get('noise', 0.0)*100}"% noise if noise > 0.0 else ""

    @staticmethod
    def __validate(props: Dict[AnyStr, Any]) -> None:
        error = []
        if props.get('n', 100) < 10 or props.get('n', 100) > 20000:
            error.append(f"n {props.get('n', 100)} should be in [10, 20000]")
        if props.get('noise', 0.1) < 0.0 or props.get('noise', 0.1) > 0.5:
            error.append(f"noise {props.get('noise', 0.15)} should be in [0, 0.5]")
        if props.get('c', 10) < 1 or props.get('c', 10) > 50:
            error.append(f"c {props.get('c', 10)} should be in [1, 50]")

        if len(error) > 0:
            raise ValueError(' - '.join(error))


class PersistentHomology(object):
    """
    Wrapper to evaluate the Persistent Homology implementation in scikit-tda - Topological Data Analysis library.
    scikit-tda leverage the Ripser library
    Documentation reference: https://docs.scikit-tda.org/en/latest/

    The data is synthetically generated from a shape (Torus, Sphere,) with additive noise.
    """
    num_shape_data_point = 48000
    size_raw_data_point = 120
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
        shape_type, raw_data = self.shaped_data_generator(props)
        props['noise'] = 0.0
        props['n'] = PersistentHomology.num_shape_data_point
        _, denoised_data = self.shaped_data_generator(props)
        return shape_type, denoised_data, raw_data

    def plot(self, kwargs: Dict[AnyStr, Any]) -> None:
        """
        Generate a 2D or 3D scatter plot with raw (noisy) data and shaped data using the enumerator ShapedDataGenerator
        @param kwargs:
        @type kwargs:
        """
        shape_type, shaped_data, raw_data = self.create_data(kwargs)
        fig = plt.figure(figsize=(8, 8))

        match self.shaped_data_generator:
            # 2D scatter plot
            case ShapedDataGenerator.CIRCLE:
                PersistentHomology.__plot2d(shape_type, shaped_data, raw_data)
            # 3D scatter plot
            case ShapedDataGenerator.TORUS | ShapedDataGenerator.SWISS_ROLL | ShapedDataGenerator.SPHERE:
                PersistentHomology.__plot3d(shaped_data, raw_data, fig)

        plt.title(shape_type)
        plt.show()

    def persistence_diagram(self, props: Dict[AnyStr, Any]) -> None:
        shape_type, data = self.shaped_data_generator(props)
        # Persistence diagrams are computation intensive so we need to limit the amount of data to be used.
        if len(data) >= 2048:
            logging.warning(f'Number of data points for persistence diagram {len(data)} truncated to 2048')
            data = data[:2048]

        # Instantiate the persistence diagram of given type
        persistence_diagram = PersistenceDiagram(data, shape_type)
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
        ax.scatter(xs=shaped_data[:, 0],
                   ys=shaped_data[:, 1],
                   zs=shaped_data[:, 2],
                   color='grey',
                   alpha=0.08,
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

