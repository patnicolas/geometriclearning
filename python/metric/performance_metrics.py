__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Dict, AnyStr, Self, List, Optional
import torch

from metric import MetricException
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from plots.plotter import Plotter, PlotterParameters
import numpy as np

"""
Wraps the various performance metrics used in training and evaluation
"""

class PerformanceMetrics(object):
    import os
    output_path = '../../'
    output_filename = 'output'
    output_folder = os.path.join(output_path, output_filename)
    valid_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    def __init__(self, metrics: Dict[MetricType, BuiltInMetric]) -> None:
        self.metrics: Dict[MetricType, BuiltInMetric] = metrics
        self.performance_values: Dict[MetricType, List[np.array]] = {}

    def show_metrics(self) -> AnyStr:
        return '\n'.join([f' {k.value}: {v}' for k, v in self.metrics.items()])

    def __str__(self) -> AnyStr:
        return '\n'.join([f' {k.value}: {v}' for k, v in self.performance_values.items()])

    @classmethod
    def build(cls, attributes: Dict[AnyStr, bool]) -> Self:
        metrics = {}
        is_weighted = attributes['class_weights'] is not None
        metrics_list = attributes['metrics_list']
        if len(metrics_list) > 0:
            for metric_label in metrics_list:
                metric_type = MetricType.get_metric_type(metric_label)
                metrics[metric_type] = BuiltInMetric(metric_type=metric_type, is_weighted=is_weighted)
        return cls(metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    def add_metric(self, metric_label: MetricType, encoding_len: int = -1, is_weighted: bool = False) -> None:
        match metric_label:
            case MetricType.Accuracy:
                self.metrics[MetricType.Accuracy] = BuiltInMetric(metric_type=MetricType.Accuracy,
                                                                  encoding_len=encoding_len,
                                                                  is_weighted=is_weighted)
            case MetricType.Precision:
                self.metrics[MetricType.Precision] = BuiltInMetric(metric_type=MetricType.Precision,
                                                                   encoding_len=encoding_len,
                                                                   is_weighted=is_weighted)
            case MetricType.Recall:
                self.metrics[MetricType.Recall] = BuiltInMetric(metric_type=MetricType.Recall,
                                                                encoding_len=encoding_len,
                                                                is_weighted=is_weighted)
            case MetricType.F1:
                self.metrics[MetricType.F1] = BuiltInMetric(metric_type=MetricType.F1,
                                                            encoding_len=encoding_len,
                                                            is_weighted=is_weighted)
            case _:
                raise MetricException(f'{str(metric_label)} is not supported')

    def collect_metrics(self, epoch: int,  metrics_value: Dict[MetricType, torch.Tensor]) -> None:
        metric_str = '\n'.join([f'   {k.value}: {v.numpy()}' for k, v in metrics_value.items()])
        print(f'>> Epoch: {epoch}\n{metric_str}')
        self.update_metrics(metrics_value)

    def update_performance_values(self, np_predicted: np.array, np_label: np.array):
        for key, metric in self.metrics.items():
            value = metric(np_predicted, np_label)
            self.update_metric(key, value)

    def update_metric(self, key: MetricType, np_value: np.array) -> None:
        if key in self.performance_values:
            values = self.performance_values[key]
            values.append(np_value)
            self.performance_values[key] = values
        else:
            values = [np_value]
            self.performance_values[key] = values

    def update_metrics(self, new_values: Dict[MetricType, torch.Tensor]) -> bool:
        """
        Update the quality metrics with new pair key-values.
        @param new_values: Set of metrics
        @type new_values: Dictionary
        """
        for key, value in new_values.items():
            if key in self.performance_values:
                values = self.performance_values[key]
                values.append(value.numpy())
                self.performance_values[key] = values
            else:
                values = [value.numpy()]
                self.performance_values[key] = values
        return len(self.performance_values.items()) > 0

    def summary(self, output_filename: AnyStr) -> None:
        """
        Plots for the various metrics and stored metrics into torch local file
        @param output_filename: Relative name of file containing the summary of metrics and losses
        @type output_filename: str
        """
        # Save the statistics in PyTorch format
        #if output_filename is not None:
        #    self.__save_summary(output_filename)

        parameters = [PlotterParameters(count=0,
                                        x_label='Epochs',
                                        y_label=k.value,
                                        title=f'{k} Plot',
                                        fig_size=(10, 6)) for idx, k in enumerate(self.performance_values.keys())]
        # Plot statistics
        attribute_values = {k.value: v for k, v in self.performance_values.items()}
        Plotter.multi_plot(attribute_values, parameters, output_filename)

    """ -------------------------------  Private Helper Methods --------------------------  """

    def __save_summary(self, output_filename) -> None:
        summary_dict = {}
        for k, lst in self.performance_values.items():
            stacked_tensor = torch.stack(lst)
            summary_dict[k] = stacked_tensor
        print(f'Save summary {str(summary_dict)}')
        torch.save(summary_dict, f"{PerformanceMetrics.output_folder}/{output_filename}.pth")

    def __record(self, epoch: int, metrics: Dict[MetricType, torch.Tensor]):
        metric_str = '\n'.join([f'   {k.value}: {v}' for k, v in metrics.items()])
        status_msg = f'>> Epoch: {epoch}\n{metric_str}'
        print(status_msg, flush=True)
        self.update_metrics(metrics)
