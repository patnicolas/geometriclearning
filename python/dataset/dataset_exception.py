__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

__all__ = ['DatasetException']

class DatasetException(Exception):
    def __init__(self, *args, **kwargs):  # real signature unknown
        super(DatasetException, self).__init__(args, kwargs)