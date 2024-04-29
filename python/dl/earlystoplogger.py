__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import Optional, AnyStr, Self, List, Dict, Tuple, NoReturn
from python.util.plotter import Plotter, PlotterParameters
import logging
logger = logging.getLogger('dl.EarlyStopLogger')

"""
    Enforce early stopping for any training/evaluation pair of execution and records loss for profiling and 
    summary
    The early stopping algorithm is implemented as follows:
       Step 1: Record new minimum evaluation loss, min_eval_loss
       Step 2: If min_eval_loss < this_eval_loss Start decreasing patience count
       Step 3: If patience count < 0, apply early stopping
    Patience for the early stop is an hyper-parameter. A metric can be optionally recorded if value is >= 0.0
"""


class EarlyStopLogger(object):
    default_min_loss = -1e-5
    train_loss_label = 'Training loss'
    eval_loss_label = "Evaluation loss"
    accuracy_label = "Accuracy"
    f1_label = "F1"
    precision_label = "Precision"
    recall_label = "Recall"

    def __init__(self,
                 patience: int,
                 min_diff_loss: Optional[float] = default_min_loss,
                 early_stopping_enabled: Optional[bool] = True):
        """
            Constructor
            @param patience: Number of time the eval_loss has been decreasing
            @type patience: int
            @param min_diff_loss: Minimum difference (min_eval_loss - lastest eval loss)
            @type min_diff_loss: float
            @param early_stopping_enabled: Early stopping is enabled if True, disabled otherwise
            @type early_stopping_enabled: bool
        """
        self.patience = patience
        self.metrics: Dict[AnyStr, List[float]] = {}
        self.min_loss = -1.0
        self.min_diff_loss = min_diff_loss
        self.early_stopping_enabled = early_stopping_enabled

    @classmethod
    def build(cls, patience: int) -> Self:
        return cls(patience, EarlyStopLogger.default_min_loss, True)

    def __call__(self, epoch: int, train_loss: float, eval_metrics: Dict[AnyStr, float] = None) -> bool:
        """
            Implement the early stop and logging of training, evaluation loss. It is assumed that at least one
            metric is provided
            @param epoch:  Current epoch index (starting with 1)
            @type epoch: int
            @param train_loss: Current training loss
            @type train_loss: float
            @param eval_loss: Current evaluation loss
            @type eval_loss: float
            @param metrics: List of pair metrics to be recorded (i.e. accuracy, precision,....)
            @type metrics: Dictionary
            @return: True if early stopping, False otherwise
            @rtype: Boolean
        """
        # Step 1. Apply early stopping criteria
        is_early_stopping = self.__evaluate(eval_metrics[EarlyStopLogger.eval_loss_label])
        # Step 2: Record training, evaluation losses and metric
        self.__record(epoch, train_loss, eval_metrics)
        logger.info(f'Is early stopping {is_early_stopping}')
        return is_early_stopping

    def update_metrics(self, metrics: Dict[AnyStr, float]) -> NoReturn:
        """
        Update the quality metrics with new pair key-values.
        @param metrics: Set of metrics
        @type metrics: Dictionary
        """
        for key, value in metrics.items():
            if key in self.metrics:
                values = self.metrics[key]
                values.append(value)
                self.metrics[key] = values
            else:
                self.metrics[key] = [value]

    def summary(self) -> NoReturn:
        """
        Plots for the various metrics
        """
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=k, fig_size=(12, 8))
                      for k, v in self.metrics.items()]
        Plotter.multi_plot(self.metrics, parameters)

    """ -----------------------  Private helper methods ----------------------  """

    def __evaluate(self, eval_loss: float) -> bool:
        is_early_stopping = False
        if self.early_stopping_enabled:
            return is_early_stopping
        else:
            if self.min_loss - eval_loss > self.min_diff_loss:
                self.min_loss = eval_loss
            else:
                self.patience -= 1

            if self.patience < 0:
                is_early_stopping = True
        return is_early_stopping

    def __record(self, epoch: int, train_loss: float, metrics: Dict[AnyStr, float]):
        metric_str = ', '.join([f'{k}: {v}' for k, v in metrics.items()])
        status_msg = f'Epoch: {epoch}, Train loss: {train_loss}, Evaluation metric: {metric_str}'
        logger.info(status_msg)
        self.update_metrics({EarlyStopLogger.train_loss_label: train_loss})
        self.update_metrics(metrics)






