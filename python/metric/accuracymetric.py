__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

import torch
import math
import numpy as np
from typing import NoReturn
from python.metric.metric import Metric


class AccuracyMetric(Metric):
    def __init__(self, eps: float):
        """
        Constructor for accuracy metric
        @param eps: Tolerance for the computation of accuracy. 0.0 for Boolean, > 0.0 for real values
        @type eps: float
        """
        super(AccuracyMetric, self).__init__()
        self.eps = eps
        self.__success = 0

    def from_tensor(self, predicted: torch.Tensor, labels: torch.Tensor) -> NoReturn:
        return self.from_numpy(predicted.numpy(), labels.numpy())

    def from_numpy(self, predicted: np.array, labels: np.array) -> NoReturn:
        """
        Compute the accuracy for prediction values defined in Numpy arrays
        @param predicted: Batch of predicted values
        @type predicted: Numpy array
        @param labels: Batch of labeled values
        @type labels: Numpy array
        """
        batch_size = len(labels)
        if batch_size > 1:
            for index, label in enumerate(labels):
                if math.fabs(predicted[index] - labels[index]) < self.eps:
                    self.__success += 1
        else:
            if math.fabs(predicted - labels) < self.eps:
                self.__success += 1
        self._count += batch_size

    def __call__(self) -> float:
        return 0.0 if self._count == 0 else float(self.__success)/self._count


