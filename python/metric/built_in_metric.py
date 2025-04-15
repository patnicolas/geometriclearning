__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
import numpy as np
from typing import List, Dict, AnyStr
from metric.metric_type import MetricType
from metric.metric import Metric
from metric import MetricException
import logging
logger = logging.getLogger('metric.BuiltInMetric')


class BuiltInMetric(Metric):
    def __init__(self, metric_type: MetricType, encoding_len: int = -1, is_weighted: bool = False):
        """
        Constructor for the accuracy metrics
        @param metric_type: Metric type (Accuracy,....)
        @type metric_type: Enumeration MetricType
        @param encoding_len: Length for the encoding (OneHot encoding)
        @type encoding_len: iny
        @param is_weighted: Specify is the precision or recall is to be weighted
        @type is_weighted: bool
        """
        super(BuiltInMetric, self).__init__()
        self.is_weighted = is_weighted
        self.metric_type = metric_type
        self.encoding_len = encoding_len

    def __str__(self) -> AnyStr:
        return f'{self.metric_type.value}, Weighted: {self.is_weighted}, Encoding length: {self.encoding_len}'

    def from_numpy(self, predicted: np.array, labeled: np.array) -> np.array:
        """
        Compute the accuracy for prediction values defined in Numpy arrays
        @param predicted: Batch of predicted values
        @type predicted: Numpy array
        @param labeled: Batch of labeled values
        @type labeled: Numpy array
        @return metric Numpy array
        @rtype Numpy array
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        if len(predicted.shape) > 2:
            raise MetricException(f'Cannot compute metric with shape {predicted.shape}')

        if self.encoding_len > 0:
            labeled = np.eye(self.encoding_len)[labeled]

        _predicted = np.argmax(predicted, axis=len(predicted.shape)-1) if len(predicted.shape) == 2 else predicted
        _labeled = np.argmax(labeled, axis=len(labeled.shape)-1) if len(labeled.shape) == 2 else labeled

        match self.metric_type:
            case MetricType.Accuracy:
                return accuracy_score(_labeled, _predicted, normalize=True, sample_weight=None) \
                    if self.is_weighted \
                    else accuracy_score(_labeled, _predicted, normalize=True)

            case MetricType.Precision:
                return precision_score(_labeled, _predicted, average=None, zero_division=1.0) if self.is_weighted \
                        else precision_score(_labeled, _predicted, average='macro', zero_division=1.0)

            case MetricType.Recall:
                return recall_score(_labeled, _predicted, average=None, zero_division=1.0) if self.is_weighted \
                        else recall_score(_labeled, _predicted, average='macro', zero_division=1.0)

            case MetricType.F1:
                return f1_score(_labeled, _predicted, average=None, zero_division=1.0) if self.is_weighted \
                        else f1_score(_labeled, _predicted, average=None, zero_division=1.0)

            case _:
                raise MetricException(f'Metric type {self.metric_type} is not supported')

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
