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
from typing import AnyStr, Union, List
import logging
# 3rd Party imports
import numpy as np
import python
from dataclasses import dataclass
__all__ = ['BoxParameter', 'FractalDimImage']


@dataclass
class BoxParameter:
    """
    Data class that wraps box parameters
    - eps: scaling factor
    - measurements: Number of measurement units (yardstick for 1D, Square in 2D, cube in 3D,..)
    """
    eps: float
    r: int

    def log_inv_eps(self) -> float:
        """
        Compute log of inverse of the scaling factor
        @return:  log(1/eps)
        @rtype: float
        """
        return -np.log(self.eps)

    def log_r(self) -> float:
        """
        Compute the log of number of measurement units
        @return: log(N)
        @rtype: float
        """
        return np.log(self.r)

    def __str__(self) -> AnyStr:
        return f'Eps: {self.eps}, Count: {self.r}'


class FractalDimImage(object):
    """
    Implementation of the computation of the Fractal dimension index of a RGB image converted into 256 grey scale
    using the box counting method.
    A fractal dimension is a measure used to describe the complexity of fractal patterns or sets by
    quantifying the ratio of change in detail relative to the change in scale.
    """
    num_grey_levels: int = 256
    max_plateau_count = 3

    def __init__(self, image_path: AnyStr) -> None:
        """
        Constructor for the computation of the fractal dimension of an image using the box counting
        method. The image is converted from RGB to grey scale if necessary
        @param image_path: Relative path of the image
        @type image_path: str
        """
        raw_image: np.array = self.__load_image(image_path)
        # If the image is actually an RGB (color) image, then converted to grey scale image
        if raw_image.shape[2] == 3:
            self.image = FractalDimImage.rgb_to_grey( raw_image)
        else:
            self.image = raw_image

    def __call__(self) -> (float, List[BoxParameter]):
        """
        Method that computes the fractal dimension of an image with a 256 grey scale.
        @return: Tuple (fractal dimension, history of box configuration
        @rtype: Tuple
        """
        image_pixels = self.image.shape[0]  # image shape
        grey_levels = FractalDimImage.num_grey_levels
        plateau_count = 0
        prev_num_r = -1  # used to check for plateaus
        trace = []
        # Leverage symmetry of fractal
        max_iters = (image_pixels // 2) + 1

        for iter in range(2, max_iters):
            # Number of box increases along the iteration  (256/n_pixels) * index
            num_boxes = grey_levels // (image_pixels // iter)
            n_boxes = max(1, num_boxes)
            num_r = 0
            eps = iter / image_pixels
            logging.info(f'{iter=}: {float(iter)/max_iters} %')

            # Populate each box with the pixels, then compute the sum of least sqiare
            for i in range(0, image_pixels, iter):
                boxes = self.__create_boxes(i, iter, n_boxes)
                num_r += FractalDimImage.__sum_least_squares(boxes, n_boxes)

            # Detect if the number of measurements has not changed... then start counting
            if num_r == prev_num_r:
                plateau_count += 1
                prev_num_r = num_r
            trace.append(BoxParameter(eps, num_r))

            # Break from the iteration if the computation is stuck in the same number of measurements
            if plateau_count > FractalDimImage.max_plateau_count:
                break
            # Implement the formula log(N)/log(eps)
        return FractalDimImage.__compute_fractal_dim(trace), trace

    @staticmethod
    def rgb_to_grey(image_array: np.array) -> np.array:
        """
        Convert an RGB image (3 channels) as a numpy array [height x width x 3] into a grey scale image
        @param image_array: Original RGB image
        @type image_array: Numpy array
        @return: Grey scale image
        @rtype: Numpy array
        """
        weights = [0.2989, 0.5870, 0.1140]
        grey_array = np.dot(image_array[..., : 3], weights)
        grey_array = np.expand_dims(grey_array, axis=-1)
        return grey_array

    """ --------------  Private Helper Methods -----------------  """

    def __create_boxes(self, i: int, iter: int, n_boxes: int) -> List[List[np.array]]:
        boxes = [[]] * ((FractalDimImage.num_grey_levels + n_boxes - 1) // n_boxes)
        i_lim = i + iter
        for row in self.image[i: i_lim]:
            for pixel in row[i: i_lim]:
                height = int(pixel // n_boxes)
                boxes[height].append(pixel)
        return boxes

    @staticmethod
    def __sum_least_squares(boxes: List[List[float]], n_boxes: int) -> float:
        # Standard deviation of boxes
        stddev_box = np.sqrt(np.var(boxes, axis=1))
        # Filter out NAN values
        stddev = stddev_box[~np.isnan(stddev_box)]

        nBox_r = 2 * (stddev // n_boxes) + 1
        return sum(nBox_r)

    @staticmethod
    def __compute_fractal_dim(trace: List[BoxParameter]) -> float:
        from numpy.polynomial.polynomial import polyfit

        _x = np.array([box_param.log_inv_eps() for box_param in trace])
        _y = np.array([box_param.log_r() for box_param in trace])
        fitted = polyfit(x=_x, y=_y, deg=1, full=False)
        return float(fitted[1])

    @staticmethod
    def __load_image(image_path: AnyStr) -> Union[np.array, None]:
        from PIL import Image
        from numpy import asarray
        try:
            this_image = Image.open(mode="r", fp=image_path)
            return asarray(this_image)
        except Exception as e:
            logging.error(f'Failed to load image {image_path}: {str(e)}')
            return None

