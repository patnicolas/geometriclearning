__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from abc import abstractmethod
from typing import List
import torch
import logging
logger = logging.getLogger('metric.Metric')

"""
Base class for all metrics
"""


class Metric(object):
    default_min_loss = -1e-5
    train_loss_label = 'Train loss'
    eval_loss_label = "Eval. loss"
    accuracy_label = "Accuracy"
    f1_label = "F1"
    precision_label = "Precision"
    recall_label = "Recall"

    def __init__(self):
        self._count = 0

    def __str__(self):
        return f'Count: {self._count}'

    @abstractmethod
    def __call__(self, predicted: List[float], labels: List[float]) -> torch.Tensor:
        raise NotImplementedError('Cannot compute an abstract metric')
