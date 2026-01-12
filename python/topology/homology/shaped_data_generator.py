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
# 3rd Party imports
import tadasets
import numpy as np
import matplotlib.pyplot as plt
__all__ = ['ShapedDataGenerator', 'ShapedDataDisplay']


@unique
class ShapedDataGenerator(Enum):
    """
    Enumerator for generation of shaped data with default values.
    The lambdas take a dictionary as input and output a tuple (Shape type, data generator)
        { } -> (Shape_type, shaped data generator)

    Example of input dictionary:
        { 'n': 250, 'noise': 0.4, 'c': 8}
    """
    size_shaped_data_point = 64000
    """ Normal Random distribution on 3D ambient space """
    NORMAL = lambda k: (np.random.randn(k.get('n', 100), 3), 'Normal Random Distribution')
    """ Uniform Random distribution on 3D ambient space """
    UNIFORM = lambda k : (np.random.rand(k.get('n', 100), 3), 'Uniform Random Distribution')
    """ Circle on 2D space """
    CIRCLE = lambda k: (tadasets.dsphere(d=1, n=k.get('n', 100), noise=k.get('noise', 0.0)),
                        ShapedDataGenerator.__title(k, 'Circle')
                        )
    """ 3D sphere """
    SPHERE = lambda k: (tadasets.sphere(n=k.get('n', 100), noise=k.get('noise', 0.0)),
                        ShapedDataGenerator.__title(k, 'Sphere')
                        )
    """  Torus """
    TORUS = lambda k: (tadasets.torus(n=k.get('n', 100),
                                      c=k.get('c', 10),
                                      a=k.get('a', 0.2),
                                      noise=k.get('noise', 0.0)),
                       ShapedDataGenerator.__title(k, 'Torus')
                       )
    """ Swiss Roll"""
    SWISS_ROLL = lambda k: (tadasets.swiss_roll(n=k.get('n', 100), noise=k.get('noise', 0.0)),
                            ShapedDataGenerator.__title(k, 'Swiss Roll')
                            )

    def __call__(self, *args, **kwargs) -> (np.array, AnyStr):
        """
        Method to return the lambda associated to a shape. The parameter values are validated prior execution of
        lambda.

        @param args: list of arguments
        @type args: List[Any]
        @param kwargs: Dictionary of arguments as key-value
        @type kwargs: Dict[AnyStr, Any]
        @return: Tuple (Shape type, shaped data)
        @rtype: Tuple
        """
        ShapedDataGenerator.__validate(kwargs)
        return self.value(*args, **kwargs)

    """ -------------------------  Private supporting methods -------------------- """

    @staticmethod
    def __title(k: Dict[AnyStr, Any], shape_type: AnyStr) -> AnyStr:
        noise = k.get('noise', 0.0)
        return f"{shape_type} with {k.get('noise', 0.0)*100}% noise" if noise > 0.0 else shape_type

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


class ShapedDataDisplay(object):
    def __init__(self, shaped_data_generator: ShapedDataGenerator) -> None:
        self.shaped_data_generator = shaped_data_generator

    def __call__(self, props: Dict[AnyStr, Any], noise: float) -> None:
        raw_data, shape_type = self.shaped_data_generator(props)
        props['noise'] = noise
        props['n'] = 96000
        shaped_data, _ = self.shaped_data_generator(props)

        fig = plt.figure(figsize=(8, 8))
        match shape_type:
            # 2D scatter plot
            case 'Circle':
                ShapedDataDisplay.__plot2d(shape_type, shaped_data, raw_data)
            # 3D scatter plot
            case 'Torus' | 'Swiss Roll' | 'Sphere':
                ShapedDataDisplay.__plot3d(shaped_data, raw_data, fig)

        plt.title(label=shape_type, fontdict={'family': 'serif', 'size': 23, 'weight': 'bold', 'color': 'blue'})
        plt.show()

    """ ------------------------  Private Supporting Methods ---------------------- """

    @staticmethod
    def __plot2d(shape_type: AnyStr, shaped_data: np.array, raw_data: np.array) -> None:
        plt.scatter(x=shaped_data[:, 0],
                    y=shaped_data[:, 1],
                    label=f'{shape_type} shaped data',
                    s=36)
        plt.scatter(x=raw_data[:, 0],
                    y=raw_data[:, 1],
                    label=f'{shape_type} raw data',
                    s=180)
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
                   alpha=0.04,
                   s=48,
                   edgecolor='none')
        ax.scatter(xs=raw_data[:, 0],
                   ys=raw_data[:, 1],
                   zs=raw_data[:, 2],
                   color='red',
                   alpha=1.0,
                   s=140,
                   edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])