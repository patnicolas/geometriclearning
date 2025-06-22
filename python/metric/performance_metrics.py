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

from typing import Dict, AnyStr, Self, List, Any
import torch

from metric import MetricException
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from plots.plotter import Plotter, PlotterParameters
import numpy as np
import logging


class PerformanceMetrics(object):
    """
    Wraps the various performance metrics used in training and evaluation. valid_metrics is the list  of
    supported metrics. A snapshot of the plots are dumped into output_folder.
    There are two constructors
    __init__: Default constructor taking a dictionary  {metric_name, build in metric} as input
            example: {metric='Accuracy', BuildInMetric(MetricType.Accuracy, is_weighted}
    build: Alternative constructor taking a dictionary   { metric_name, is_weighted} as input
            example: {metric='Accuracy', is_weighted=True}
    """
    import os
    output_path = '../../'
    output_filename = 'output'
    output_folder = os.path.join(output_path, output_filename)
    valid_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    def __init__(self, metrics: Dict[MetricType, BuiltInMetric]) -> None:
        """
        Default constructor for the Performance metrics taking a dictionary input a
        { metric_name, build in metric}   example {metric='Accuracy', BuildInMetric(MetricType.Accuracy, is_weighted}
        @param metrics: Dictionary { metric_name, built in metric}
        @type metrics: Dictionary
        """
        self.metrics: Dict[MetricType, BuiltInMetric] = metrics
        self.performance_values: Dict[MetricType, List[np.array]] = {}

    def show_metrics(self) -> AnyStr:
        return '\n'.join([f' {k.value}: {str(v)}' for k, v in self.metrics.items()])

    def __str__(self) -> AnyStr:
        return '\n'.join([f' {k.value}: {str(lst)}' for k, lst in self.performance_values.items()])

    @classmethod
    def build(cls, metric_attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor for the Performance metrics taking a dictionary input as
         { metric_name, is_weighted}  example {metric='Accuracy', is_weighted=True}
        @param metric_attributes: Dictionary { metric_name, is_weighted}
        @type metric_attributes: Dictionary
        @return: Instance of PerformanceMetrics
        @rtype: PerformanceMetrics
        """
        metrics_list = metric_attributes['metrics_list']
        is_weighted = metric_attributes['is_class_imbalance']
        metrics = {MetricType.get_metric_type(metric): BuiltInMetric(metric_type=MetricType.get_metric_type(metric),
                                                                     is_weighted=is_weighted)
                   for metric in metrics_list}
        return cls(metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    def add_metric(self, metric_label: MetricType, encoding_len: int = -1, is_weighted: bool = False) -> None:
        """
        Add or register a new performance metric such as Precision, Recall, ...
        @param metric_label: Name for the metric (i.e. 'Accuracy')
        @type metric_label: str
        @param encoding_len: Length for the encoding (ie. output Number of classes) with default -1 (no encoding)
        @type encoding_len: int
        @param is_weighted: Specify if the metric is weighted for class distribution imbalance
        @type is_weighted: bool
        """
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
        logging.info(f'>> Epoch: {epoch}\n{metric_str}')
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
        logging.info(f'Save summary {str(summary_dict)}')
        torch.save(summary_dict, f"{PerformanceMetrics.output_folder}/{output_filename}.pth")

    def __record(self, epoch: int, metrics: Dict[MetricType, torch.Tensor]):
        metric_str = '\n'.join([f'   {k.value}: {v}' for k, v in metrics.items()])
        status_msg = f'>> Epoch: {epoch}\n{metric_str}'
        logging.info(status_msg, flush=True)
        self.update_metrics(metrics)
