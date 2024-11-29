__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from enum import Enum


class MetricType(Enum):
    Accuracy = 0
    Precision = 1
    Recall = 2
    F1 = 3
