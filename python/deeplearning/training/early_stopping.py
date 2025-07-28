__author__ = "Patrick R. Nicolas"
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

from typing import Optional, Dict, Any, AnyStr, Self
from metric.metric import Metric
__all__ = ['EarlyStopping']


class EarlyStopping(object):
    """
    Wraps the early stopping condition. An early stop is triggers when the loss function increases
    by  >  'min_diff_loss' for 'patience' number of consecutive times.
    Example: Given a min_diff_loss = 3% and patience = 2, a loss function decreasing for 23 epochs then increasing
    by 5% 3 times in the row will trigger an early stop
    """
    def __init__(self,
                 patience: int,
                 min_diff_loss: Optional[float] = Metric.default_min_loss) -> None:
        """
        Constructor for the Early stopping mechanism

        @param patience: Number of times the loss should be within limits before exiting training
        @type patience: int
        @param min_diff_loss: Optional minimum value of increase of loss to trigger early stopping
        @type min_diff_loss: float
        """
        assert 1 <= patience <= 10, f'Patience for early stopping {patience} should be [1, 10]'

        self.patience = patience
        self.min_diff_loss = min_diff_loss

    @classmethod
    def build(cls, attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor for the early stopping criteria using a dictionary of attributes

        @param attributes: Dictionary of the attributes for early stopping
        @type attributes: Dictionary [str, parameters]
        @return: Instance of EarlyStopping
        @rtype: EarlyStopping
        """
        patience = attributes.get('patience', 2)
        min_diff_loss = attributes.get('min_diff_loss', Metric.default_min_loss)
        return cls(patience, min_diff_loss)

    def __str__(self) -> AnyStr:
        return f'\n   Patience:             {self.patience}\n   Min. difference loss: {self.min_diff_loss}'