__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

from abc import abstractmethod

"""
Base class for all metrics
"""


class Metric(object):
    def __init__(self):
        self._count = 0

    def __str__(self):
        return f'Count: {self._count}'

    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError('Cannot compute an abstract metric')