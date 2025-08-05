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
from enum import Enum
from typing import AnyStr, Self
# Library imports
from metric import MetricException
__all__ = ['MetricType']


class MetricType(Enum):
    Accuracy = 'Accuracy'
    Precision = 'Precision'
    Recall = 'Recall'
    F1 = 'F1'
    AuROC = 'AuROC'
    TrainLoss = 'TrainLoss'
    EvalLoss = 'EvalLoss'
    All = 'All'

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
            case 'EvalLoss':
                return MetricType.EvalLoss
            case 'TrainLoss':
                return MetricType.TrainLoss
            case 'All':
                return MetricType.All
            case _:
                raise MetricException(f'{metric_type_str} metric is not supported')
