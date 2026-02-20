__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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
from abc import abstractmethod
from typing import List
# 3rd Party imports
import torch
__all__ = ['Metric']


class Metric(object):
    """
    Base, abstract class for all metrics
    """
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
