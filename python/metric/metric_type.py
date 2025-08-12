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
from enum import StrEnum
from typing import AnyStr
import logging
# Library imports
from metric import MetricException
import python
__all__ = ['MetricType', 'get_metric_type']


class MetricType(StrEnum):
    Accuracy = 'Accuracy'
    Precision = 'Precision'
    Recall = 'Recall'
    F1 = 'F1'
    AucROC = 'AucROC'
    AucPR = 'AucPR'
    Jaccard = 'Jaccard'
    TrainLoss = 'TrainLoss'
    EvalLoss = 'EvalLoss'
    All = 'All'


_lookup = {item.value: item for item in MetricType}

def get_metric_type(metric_type_str: AnyStr) -> MetricType:
    try:
        return _lookup[metric_type_str]
    except KeyError as e:
        logging.error(e)
        raise MetricException(e)
