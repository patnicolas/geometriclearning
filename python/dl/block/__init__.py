__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Tuple

Conv2DataType = Tuple[int, int]
Conv3DataType = Tuple[int, int, int]
ConvDataType = int | Conv2DataType | Conv3DataType
