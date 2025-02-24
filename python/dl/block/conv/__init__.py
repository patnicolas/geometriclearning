__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Tuple

""" Define the types for the 3 dimension of convolutional networks """
Conv1DataType = int
Conv2DataType = Tuple[int, int]
Conv3DataType = Tuple[int, int, int]
ConvDataType = Conv1DataType | Conv2DataType | Conv3DataType