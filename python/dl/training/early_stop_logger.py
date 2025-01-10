__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Optional, AnyStr, Self, List, Dict, NoReturn
from plots.plotter import Plotter, PlotterParameters
from metric.metric import Metric
import torch
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
    output_folder = '../../../../tests/output'

    def __init__(self,
                 patience: int,
                 min_diff_loss: Optional[float] = Metric.default_min_loss,
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
        self.metrics: Dict[AnyStr, List[torch.Tensor]] = {}
        self.min_loss = -1.0
        self.min_diff_loss = min_diff_loss
        self.early_stopping_enabled = early_stopping_enabled

    @classmethod
    def build(cls, patience: int) -> Self:
        """
        Alternative, simplified constructor with early stopping enabled and default minimum
        loss for convergence during training
        @param patience: Frequency/number of times the loss is below the minimum loss
        @type patience: int
        @return: New instance of EarlyStopLogger
        @rtype: EarlyStopLogger
        """
        return cls(patience, Metric.default_min_loss, early_stopping_enabled=True)

    def __call__(self, epoch: int, train_loss: torch.Tensor, eval_metrics: Dict[AnyStr, torch.Tensor] = None) -> bool:
        """
            Implement the early stop and logging of training, evaluation loss. It is assumed that at least one
            metric is provided
            @param epoch:  Current epoch index (starting with 1)
            @type epoch: int
            @param train_loss: Current training loss
            @type train_loss: float
            @param eval_metrics: List of pair metrics to be recorded (i.e. accuracy, precision,....)
            @type eval_metrics: Dictionary
            @return: True if early stopping, False otherwise
            @rtype: Boolean
        """
        # Step 1. Apply early stopping criteria
        loss_value = eval_metrics[Metric.eval_loss_label]
        # is_early_stopping = self.__evaluate(torch.Tensor(loss_value))
        # Step 2: Record training, evaluation losses and metric
        self.__record(epoch, train_loss, eval_metrics)
        # print(f'Is early stopping {is_early_stopping}', flush=True)
        return False

    def update_metrics(self, metrics: Dict[AnyStr, float]) -> bool:
        """
        Update the quality metrics with new pair key-values.
        @param metrics: Set of metrics
        @type metrics: Dictionary
        """
        for key, value in metrics.items():
            if key in self.metrics:
                values = self.metrics[key]
                values.append(torch.Tensor(value))
                self.metrics[key] = values
            else:
                values = [torch.Tensor(value)]
                self.metrics[key] = values
        return len(self.metrics.items()) > 0

    def summary(self, output_filename: Optional[AnyStr] = None) -> None:
        """
        Plots for the various metrics and stored metrics into torch local file
        @param output_filename: Relative name of file containing the summary of metrics and losses
        @type output_filename: str
        """
        y_labels = ['Accuracy', 'Precision', 'Recall', 'Training Loss', 'Evaluation Loss']
        for idx, k in enumerate(self.metrics.keys()):
            print(k)
        parameters = [PlotterParameters(0, x_label='Iteration', y_label=k, title=f'{k} Plot', fig_size=(12, 8))
                      for idx, k in enumerate(self.metrics.keys())]

        # Save the statistics in PyTorch format
        # if output_filename is not None:
        #    self.__save_summary(output_filename)
        # Plot statistics
        Plotter.multi_plot(self.metrics, parameters, output_filename)

    @staticmethod
    def load_summary(path_name: AnyStr, summary_filename: AnyStr) -> Dict[AnyStr, List[torch.Tensor]]:
        """
        Load the content of a dictionary of list of torch tensor from a given output file
        @param path_name: Relative path of the file containing the summary performance stats
        @type path_name: str
        @param summary_filename: Name of the file containing the summary performance stats
        @type summary_filename: str
        @return: Return the summary of performance statistics as a dictionary of metrics, list of tensors
        @rtype: Dict[AnyStr, List[torch.Tensor]]
        """
        stacked_tensor_dict = torch.load(f"{path_name}/{summary_filename}.pth")
        tensor_dict = {}
        for k, stacked_tensor in stacked_tensor_dict.items():
            tensor_dict[k] = list(torch.unbind(stacked_tensor, dim=0))
        return tensor_dict

    """ -----------------------  Private helper methods ----------------------  """

    def __evaluate(self, eval_loss: torch.Tensor) -> bool:
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
        print(status_msg, flush=True)
        metrics[Metric.train_loss_label] = train_loss
        self.update_metrics(metrics)

    def __save_summary(self, output_filename) -> NoReturn:
        summary_dict = {}
        for k, lst in self.metrics.items():
            stacked_tensor = torch.stack(lst)
            summary_dict[k] = stacked_tensor
        torch.save(summary_dict, f"{EarlyStopLogger.output_folder}/{output_filename}.pth")







