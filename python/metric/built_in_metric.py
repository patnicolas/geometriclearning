__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import List, AnyStr, Dict
# 3rd Party imports
import torch
import numpy as np
# Library imports
from metric.metric_type import MetricType
from metric.metric import Metric
from metric import MetricException
__all__ = ['BuiltInMetric']


class BuiltInMetric(Metric):

    def __init__(self, metric_type: MetricType,
                 encoding_len: int = -1,
                 is_weighted: bool = False,
                 is_multi_class: bool = True):
        """
        Constructor for the accuracy metrics
        @param metric_type: Metric type (Accuracy,....)
        @type metric_type: Enumeration MetricType
        @param encoding_len: Length for the encoding (OneHot encoding)
        @type encoding_len: iny
        @param is_weighted: Specify is the precision or recall is to be weighted
        @type is_weighted: bool
        @param is_multi_class: Boolean flag to specify if these are multi-class labels
        @type is_multi_class: bool
        """
        super(BuiltInMetric, self).__init__()
        self.is_weighted = is_weighted
        self.metric_type = metric_type
        self.encoding_len = encoding_len
        self.is_multi_class = is_multi_class

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
        if len(predicted.shape) > 2:
            raise MetricException(f'Cannot compute metric with shape {predicted.shape}')
        if self.encoding_len > 0:
            labeled = np.eye(self.encoding_len)[labeled]

        match self.metric_type:
            case MetricType.Accuracy:
                return self.__accuracy(labeled, predicted)

            case MetricType.Precision:
                return self.__precision(labeled, predicted)

            case MetricType.Recall:
                return self.__recall(labeled, predicted)

            case MetricType.F1:
                return self.__f1(labeled, predicted)

            case MetricType.AucROC:
                return self.__auc_roc_score(labeled, predicted)

            case MetricType.AucPR:
                return self.__auc_pr_score(labeled, predicted)

            case _:
                raise MetricException(f'Metric type {self.metric_type} is not supported')

    def from_float(self, predicted: List[float], labels: List[float]) -> float:
        """
           Compute a given metric for predicted values defined in float values.

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
        return np_metric

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

    def __call__(self, predicted: np.array, labeled: np.array) -> np.array:
        value = self.from_numpy(predicted, labeled)
        return np.array(value)

    def default(self, predicted: List[float], labeled: List[float]) -> Dict[MetricType, torch.Tensor]:
        """
        Generic computation of a metric
        @param predicted: Batch of predicted values
        @type predicted: List[float]
        @param labeled: Batch of labeled values
        @type labeled:List[float]
        @return: One or multiple metric values
        @rtype: Torch Tensor
        """
        if len(predicted) != len(labeled):
            raise MetricException(f'Number of predicted values {len(predicted)} != Number of labels {len(labeled)}')

        _predicted = np.array(predicted)
        _labeled = np.array(labeled)

        return {
            MetricType.Accuracy: torch.from_numpy(self.__accuracy(_labeled, _predicted)),
            MetricType.Precision: torch.from_numpy(self.__precision(_labeled, _predicted)),
            MetricType.Recall: torch.from_numpy(self.__recall(_labeled, _predicted)),
            MetricType.F1: torch.from_numpy(self.__f1(_labeled, _predicted))
        }

    """ ------------------------  Private methods -------------------------  """

    def __accuracy(self, _labeled: np.array, _predicted: np.array) -> np.array:
        from sklearn.metrics import accuracy_score
        labeled, predicted = BuiltInMetric.__get_class_prediction(_labeled, _predicted)
        score = accuracy_score(labeled, predicted, normalize=True)
        return np.array([score])

    def __precision(self, _labeled: np.array, _predicted: np.array) -> np.array:
        from sklearn.metrics import precision_score
        labeled, predicted = BuiltInMetric.__get_class_prediction(_labeled, _predicted)
        score = precision_score(y_true=labeled, y_pred=predicted, average='macro', zero_division=1.0)
        return np.array([score])

    def __recall(self, _labeled: np.array, _predicted: np.array) -> np.array:
        from sklearn.metrics import recall_score
        labeled, predicted = BuiltInMetric.__get_class_prediction(_labeled, _predicted)
        score = recall_score(y_true=labeled,
                             y_pred=predicted,
                             average='macro',
                             zero_division=1.0)
        return np.array([score])

    def __f1(self, _labeled: np.array, _predicted: np.array) -> np.array:
        precision = self.__precision(_labeled, _predicted)
        recall = self.__recall(_labeled, _predicted)
        return 2.0*precision*recall/(precision + recall)

    def __auc_roc_score(self, _labeled: np.array, _predicted: np.array) -> np.array:
        from sklearn.metrics import roc_auc_score
        # One vs rest AUC
        _labeled_bin = BuiltInMetric.__get_labeled_classes(_labeled)
        return roc_auc_score(y_true=_labeled_bin,
                             y_score=_predicted,
                             average='macro' if self.is_weighted else None,
                             multi_class='ovr' if self.is_multi_class else 'raise')

    def __auc_pr_score(self, _labeled: np.array, _predicted: np.array) -> np.array:
        from sklearn.metrics import average_precision_score
        # One vs rest AUC
        _labeled_bin = BuiltInMetric.__get_labeled_classes(_labeled)
        return average_precision_score(y_true=_labeled_bin,
                                       y_score=_predicted,
                                       average='macro' if self.is_weighted else None)

    def __jaccard(self, _labeled: np.array, _predicted: np.array) -> np.array:
        from sklearn.metrics import jaccard_score

        return jaccard_score(_labeled, _predicted, average='macro', zero_division=1.0) if self.is_weighted \
            else jaccard_score(_labeled, _predicted, average=None, zero_division=1.0)

    @staticmethod
    def __get_labeled_classes(_labeled: np.array) -> np.array:
        from sklearn.preprocessing import label_binarize
        max_class = np.max(_labeled)
        _classes = [0] + list(range(1, max_class + 1))
        return label_binarize(_labeled, classes=_classes)

    @staticmethod
    def __get_class_prediction(labeled: np.array, predicted: np.array) -> (np.array, np.array):
        _predicted = np.argmax(predicted, axis=len(predicted.shape) - 1) if len(predicted.shape) == 2 else predicted
        _labeled = np.argmax(labeled, axis=len(labeled.shape) - 1) if len(labeled.shape) == 2 else labeled
        return _labeled, _predicted
