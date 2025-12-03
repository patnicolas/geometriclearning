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
from dataclasses import dataclass
import logging
from typing import List, Optional, AnyStr, Dict, AnyStr, Any
from enum import Enum, unique
# 3rd Party imports
import tadasets
import numpy as np
import matplotlib.pyplot as plt
import python


@unique
class HomologyShape(Enum):
    CIRCLE = lambda kwargs: ('CIRCLE', tadasets.dsphere(d=1, n=kwargs.get('n', 100), noise=kwargs.get('noise', 0.0)))
    SPHERE = lambda kwargs: ('SPHERE', tadasets.sphere(n=kwargs.get('n', 100), noise=kwargs.get('noise', 0.0)))
    TORUS = lambda kwargs: ('TORUS', tadasets.torus(n=kwargs.get('n', 100),
                                                    c=kwargs.get('c', 10),
                                                    a=kwargs.get('a', 0.2),
                                                    noise=kwargs.get('noise', 0.0)))

    def __call__(self, *args, **kwargs) -> (AnyStr, np.array):
        return self.value(*args, **kwargs)



class PersistentHomology(object):
    def __init__(self, homology_shape: HomologyShape) -> None:
        self.homology_shape = homology_shape

    def create_data(self, kwargs: Dict[AnyStr, Any]) -> (AnyStr, np.array, np.array):
        shape_type, raw_data = self.homology_shape(kwargs)
        kwargs['noise'] = 0.0
        kwargs['n'] = 8000
        _, denoised_data = self.homology_shape(kwargs)
        return shape_type, denoised_data, raw_data

    def plot(self, kwargs: Dict[AnyStr, Any]) -> None:
        shape_type, shape_data, raw_data = self.create_data(kwargs)
        fig = plt.figure(figsize=(8, 8))

        match self.homology_shape:
            case HomologyShape.CIRCLE:
                PersistentHomology.__plot2d(shape_type, shape_data, raw_data)
            case HomologyShape.TORUS:
                PersistentHomology.__plot3d(shape_data, raw_data, fig)
            case HomologyShape.SPHERE:
                PersistentHomology.__plot3d(shape_data, raw_data, fig)
        plt.show()

    @staticmethod
    def __plot2d(shape_type: AnyStr, shape_data: np.array, raw_data: np.array) -> None:
        plt.scatter(shape_data[:, 0], shape_data[:, 1], label='original shape', s=16)
        plt.scatter(raw_data[:, 0], raw_data[:, 1], label='noisy shape', s=48)
        plt.title(shape_type)
        plt.legend()

    @staticmethod
    def __plot3d(shape_data: np.array, raw_data: np.array, fig) -> None:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(shape_data[:, 0], shape_data[:, 1], shape_data[:, 2], color='grey', alpha=0.5, s=20, edgecolor='none')
        ax.scatter(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2], color='red', alpha=0.5, s=100,  edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])







if __name__ == '__main__':
    num_raw_points = 260
    kwargs = {'n': num_raw_points}
    persistent_homology = PersistentHomology(HomologyShape.TORUS)
    persistent_homology.plot(kwargs)
    kwargs['n'] = num_raw_points
    persistent_homology = PersistentHomology(HomologyShape.SPHERE)
    persistent_homology.plot(kwargs)
    """
    persistent_homology.plot3d(kwargs_1)

    persistent_homology = PersistentHomology(HomologyShape.TORUS)
    persistent_homology.plot3d(kwargs_1)
    """
