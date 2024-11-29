__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import numpy as np
from typing import Self, AnyStr, List

"""
Class that encapsulates the computation of the Fractal Dimension of an 3D object.
The implementation relies on the simple box counting method for which the size 
is incrementally decrease and the number of counts increased.
For simplicity sake, we assume that the object can be fully embedded in a cube.

The class has two constructors:
- Default constructor with a given 3D Numpy array and a threshold value
- Alternative constructor, build, for which the numpy array is created with a given,
      number of values, identical for each of the 3 dimension, x, y and z.
"""


class FractalDimObject(object):
    def __init__(self, xyz: np.array, threshold: float) -> None:
        """
        Default constructor for the computation of the Fractal
        @param xyz: Input data as 3-dimension numpy array
        @type xyz: np array
        @param threshold: Threshold used to extract the counting boxes
        @type threshold: float
        """
        assert len(xyz.shape) == 3, f'The shape of data {xyz.shape} should be 3'
        assert 0.8 <= threshold < 1.0, f'Threshold {threshold} should be [0.8, 1,0['

        self.xyz = xyz
        self.threshold = threshold

    @classmethod
    def build(cls, size: int, threshold: float) -> Self:
        """
        Alternative constructor for the computation of the Fractal dimension of an object
        @param size: Number of X, Y and Z identical values
        @type size: int
        @param threshold: Threshold used to extract the counting boxes
        @type threshold: float
        @return: Instance of the Fractal Dimension object class
        @rtype: FractalDimObject
        """
        import random

        assert 0.8 <= threshold < 1.0, f'Threshold {threshold} should be [0.8, 1,0['

        _xyz = np.zeros((size, size, size))
        # Create a 3D fractal-like structure such as cube
        for x in range(size):         # Width
            for y in range(size):     # Depth
                for z in range(size): # Height
                    if (x // 2 + y // 2) % 2 == 0:
                        mean = size//2
                        std_dev = size
                        _xyz[x, y, z] = random.gauss(mean, std_dev)

        return cls(_xyz, threshold)

    def __call__(self) -> (np.array, List[int], List[int]):
        """
        Implement the computation of the counting of box used to completely cover
        a 3D object.
        @return: Tuple (fractal dimension, array of sizes, array of counts)
        @rtype: Tuple (np.array, int, int)
        """
        # Step 1 Extract the sizes of array
        sizes = self.__extract_size()
        sizes_list = list(sizes)

        # Step 2 Count the number of boxes of each size
        counts = [self.__count_boxes(int(size)) for size in sizes_list]

        # Step 3 Fit the points to a line
        coefficients = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coefficients[0], sizes, counts

    def __str__(self) -> AnyStr:
        return f'Input data:\n{str(self.xyz)}'

    """ --------------  Supporting Helper Methods ---------------------- """

    def __extract_size(self) -> np.array:
        # Remove values close to 1.0
        filtered = (self.xyz < self.threshold)
        # Minimal dimension of box size
        min_dim = min(filtered.shape)
        # Greatest power of 2 less than or equal to p
        n = 2 ** np.floor(np.log(min_dim) / np.log(2))
        # Extract the sizes
        size_x: int = int(np.log(n) / np.log(2))
        return np.arange(size_x, 1, -1) * 2

    def __count_boxes(self, box_size: int) -> int:
        sx, sy, sz = self.xyz.shape
        count = 0
        for i in range(0, sz, box_size):
            for j in range(0, sy, box_size):
                for k in range(0, sz, box_size):
                    data = self.xyz[i:i+box_size, j:j+box_size, k:k+box_size]
                    if np.any(data):
                        count += 1
        return count
