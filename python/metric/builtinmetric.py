__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."


from python.metric.metric import Metric
import torch
import numpy as np
from enum import Enum
from typing import List


class MetricType(Enum):
    Accuracy = 0
    Precision = 1
    Recall = 2


class BuiltInMetric(Metric):
    def __init__(self, metric_type: MetricType, is_weighted: bool = False):
        """
        Constructor for accuracy metric
        @param metric_type: Metric type (Accuracy,....)
        @type metric_type: Enumeration MetricType
        @param is_weighted: Specify is the precision or recall is to be weighted
        @type bool
        """
        super(BuiltInMetric, self).__init__()
        self.is_weighted = is_weighted
        self.metric_type = metric_type

    def from_numpy(self, predicted: np.array, labels: np.array) -> np.array:
        """
        Compute the accuracy for prediction values defined in Numpy arrays
        @param predicted: Batch of predicted values
        @type predicted: Numpy array
        @param labels: Batch of labeled values
        @type labels: Numpy array
        """
        match self.metric_type:
            case MetricType.Accuracy: return BuiltInMetric.__accuracy(predicted, labels)
            case MetricType.Precision: return self.__precision(predicted, labels)
            case MetricType.Recall:return self.__recall(predicted, labels)

    def __call__(self, predicted: List[float], labels: List[float]) -> torch.Tensor:
        # Need transfer prediction and labels to CPU for using numpy
        np_metric = self.from_numpy(predicted, labels)
        return torch.tensor(np_metric)

    """ ----------------------------  Private Helper Methods ---------------------- """
    def __precision(self, predicted: np.array, labels: np.array) -> np.array:
        from sklearn.metrics import precision_score
        _predicted = np.where(predicted > 0.5, 1.0, 0.0)
        return precision_score(labels, _predicted, average="weighted") if self.is_weighted \
            else precision_score(labels, _predicted)

    def __recall(self, predicted: np.array, labels: np.array) -> np.array:
        from sklearn.metrics import recall_score
        _predicted = np.where(predicted > 0.5, 1.0, 0.0)
        return recall_score(labels, _predicted, average="weighted") if self.is_weighted \
            else recall_score(labels, _predicted)

    @staticmethod
    def __accuracy(predicted: np.array, labels: np.array) -> np.array:
        from sklearn.metrics import accuracy_score
        _predicted = np.where(predicted > 0.5, 1.0, 0.0)
        return accuracy_score(labels, _predicted, normalize=True)