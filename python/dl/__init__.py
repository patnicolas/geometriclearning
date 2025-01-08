__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Tuple

Conv2DataType = Tuple[int, int]
Conv3DataType = Tuple[int, int, int]
ConvDataType = int | Conv2DataType | Conv3DataType


class DLException(Exception):
    def __init__(self, *args, **kwargs):
        super(DLException, self).__init__(args, kwargs)


class ConvException(DLException):
    def __init__(self, *args, ** kwargs):
        super(ConvException, self).__init__(args, kwargs)


class VAEException(DLException):
    def __init__(self, *args, **kwargs):
        super(VAEException, self).__init__(args, kwargs)


class GNNException(DLException):
    def __init__(self, *args, **kwargs):
        super(GNNException, self).__init__(args, kwargs)


class TrainingException(DLException):
    def __init__(self, *args, **kwargs):
        super(TrainingException, self).__init__(args, kwargs)


class ValidationException(DLException):
    def __init__(self, *args, **kwargs):
        super(ValidationException, self).__init__(args, kwargs)