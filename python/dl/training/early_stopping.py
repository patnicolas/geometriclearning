__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Optional, Dict, Any, AnyStr, Self
from metric.metric import Metric

"""
Wraps the early stopping condition. An early stop is triggers when the loss function increases
by  >  'min_diff_loss' for 'patience' number of consecutive times.
Example: Given a min_diff_loss = 3% and patience = 2, a loss function decreasing for 23 epochs then increasing 
by 5% 3 times in the row will trigger an early stop


with the following parameters
- min_diff_loss: Minimum value of increase of loss to trigger early stopping
- patience: Number of times the loss increases before triggering early stopping
"""

class EarlyStopping(object):
    def __init__(self,
                 patience: int,
                 min_diff_loss: Optional[float] = Metric.default_min_loss) -> None:
        self.patience = patience
        self.min_diff_loss = min_diff_loss

    @classmethod
    def build(cls, attributes: Dict[AnyStr, Any]) -> Self:
        patience = attributes.get('patience', 2)
        min_diff_loss = attributes.get('min_diff_loss', Metric.default_min_loss)
        return cls(patience, min_diff_loss)

    def __str__(self) -> AnyStr:
        return f'\n   Patience:             {self.patience}\n   Min. difference loss: {self.min_diff_loss}'