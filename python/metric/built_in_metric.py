__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
import numpy as np
from typing import List, Dict, AnyStr
from metric.metric_type import MetricType
from metric.metric import Metric
import logging
logger = logging.getLogger('metric.BuiltInMetric')


class BuiltInMetric(Metric):
    def __init__(self, metric_type: MetricType, encoding_len: int, is_weighted: bool = False):
        """
        Constructor for the accuracy metrics
        @param metric_type: Metric type (Accuracy,....)
        @type metric_type: Enumeration MetricType
        @param is_weighted: Specify is the precision or recall is to be weighted
        @type is_weighted: bool
        """
        super(BuiltInMetric, self).__init__()
        self.is_weighted = is_weighted
        self.metric_type = metric_type
        self.encoding_len = encoding_len

    def __repr__(self) -> AnyStr:
        return f'Metric: {self.metric_type} weighted: {self.is_weighted} Encoded: {self.encoding_len}'

    def from_numpy(self, predicted: np.array, labels: np.array) -> np.array:
        """
        Compute the accuracy for prediction values defined in Numpy arrays
        @param predicted: Batch of predicted values
        @type predicted: Numpy array
        @param labels: Batch of labeled values
        @type labels: Numpy array
        @return metric Numpy array
        @rtype Numpy array
        """
        if self.encoding_len > 0:
            labels = np.eye(self.encoding_len)[labels]

        match self.metric_type:
            case MetricType.Accuracy: return self.__accuracy(predicted, labels)
            case MetricType.Precision: return self.__precision(predicted, labels)
            case MetricType.Recall: return self.__recall(predicted, labels)
            case MetricType.F1: return self.__f1(predicted, labels)

    def from_float(self, predicted: List[float], labels: List[float]) -> float:
        """
           Compute the accuracy for prediction values defined in float values
           @param predicted: Batch of predicted values
           @type predicted: List of floats
           @param labels: Batch of labeled values
           @type labels: List of float
           @return metric value
           @rtype float
           """

        assert len(predicted) == len(labels), \
            f'Number of prediction {len(predicted)} != Number of labels {len(labels)}'

        np_predicted = np.array(predicted)
        np_labels = np.array(labels)
        np_metric = self.from_numpy(np_predicted, np_labels)
        return float(np_metric)

    def from_torch(self, predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the accuracy for prediction values defined in Torch tensor
        @param predicted: Batch of predicted values
        @type predicted: Torch tensor
        @param labels: Batch of labeled values
        @type labels: Torch tensor
        @return metric tensor
        @rtype Torch tensor
        """
        assert len(predicted) == len(labels), \
            f'Number of prediction {len(predicted)} != Number of labels {len(labels)}'

        np_predicted = predicted.numpy()
        np_labels = labels.numpy()
        np_metric = self.from_numpy(np_predicted, np_labels)
        return torch.tensor(np_metric)

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

    def __accuracy(self, predicted: np.array, labels: np.array) -> np.array:
        from sklearn.metrics import accuracy_score

        _predicted = np.where(predicted > 0.5, 1.0, 0.0)
        return accuracy_score(labels, _predicted, normalize=True, sample_weight="weighted") if self.is_weighted \
            else accuracy_score(labels, _predicted, normalize=True)

    def __f1(self, predicted: np.array, labels: np.array) -> np.array:
        from sklearn.metrics import f1_score

        _predicted = np.where(predicted > 0.5, 1.0, 0.0)
        return f1_score(labels, _predicted, average="weighted") if self.is_weighted \
            else f1_score(labels, _predicted)


def create_metric_dict(metric_labels: List[AnyStr], encoding_len: int) -> Dict[AnyStr, BuiltInMetric]:
    metrics = {}
    assert metric_labels is not None and len(metric_labels) > 0

    for metric_label in metric_labels:
        match metric_label:
            case Metric.accuracy_label:
                metrics[Metric.accuracy_label] = BuiltInMetric(MetricType.Accuracy,
                                                               encoding_len=encoding_len,
                                                               is_weighted=True)
            case Metric.precision_label:
                metrics[Metric.precision_label] = BuiltInMetric(MetricType.Precision,
                                                                encoding_len=encoding_len,
                                                                is_weighted=True)
            case Metric.recall_label:
                metrics[Metric.recall_label] = BuiltInMetric(MetricType.Recall,
                                                             encoding_len=encoding_len,
                                                             is_weighted=True)
            case Metric.f1_label:
                metrics[Metric.f1_label] = BuiltInMetric(MetricType.F1,
                                                         encoding_len=encoding_len,
                                                         is_weighted=True)
            case _:
                logger.warning(f'Metric {metric_label} is not supported')
    return metrics
