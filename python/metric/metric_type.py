__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from enum import Enum
from typing import AnyStr, Self

from metric import MetricException


class MetricType(Enum):
    Accuracy = 'Accuracy'
    Precision = 'Precision'
    Recall = 'Recall'
    F1 = 'F1'
    AuROC = 'AuROC'
    TrainLoss = 'TrainLoss'
    EvalLoss = 'EvalLoss'

    @staticmethod
    def get_metric_type(metric_type_str: AnyStr) -> Self:
        match metric_type_str:
            case 'Accuracy':
                return MetricType.Accuracy
            case 'Precision':
                return MetricType.Precision
            case 'Recall':
                return MetricType.Recall
            case 'F1':
                return MetricType.F1
            case _:
                raise MetricException(f'{metric_type_str} metric is not supported')
