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
import logging
from typing import Dict, AnyStr, Self, List, Optional
import os
# 3rd Party imports
import torch
import numpy as np
# Library imports
from metric import MetricException
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from plots.metric_plotter import MetricPlotter, MetricPlotterParameters
import python
__all__ = ['PerformanceMetrics']


class PerformanceMetrics(object):
    """
    Wraps the various performance metrics used in training and evaluation. valid_metrics is the list  of
    supported metrics. A snapshot of the plots are dumped into output_folder.
    The performance metrics are categorized as
    - registered metrics: Performance metrics available for training
    - current metrics: Performance metrics actually collected during training of a given model

    There are two constructors
    __init__: Default constructor taking a dictionary  {metric_name, build in metric} as input
            The default constructor does not automatically include the training and validation losses
            Example: {metric='Accuracy', BuildInMetric(MetricType.Accuracy, is_weighted}

    build: Alternative constructor taking a list of metric labels and flag to specify if the
            class are imbalance.
            This alternative constructor will automatically add the training and validation losses if they
            are not explicitly defined
            example: metrics_list=['Accuracy', 'f1'],  is_class_imbalance=True

    Attributes:
        registered_perf_metrics: Dictionary of the metric registered for this model
        collected_metrics: Dictionary of the metric currently collected for this training session
    """
    perf_metrics_directory = '../../perf_metrics/'
    if not os.path.exists(perf_metrics_directory):
        os.makedirs(perf_metrics_directory)

    valid_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR']

    def __init__(self,
                 registered_perf_metrics: Dict[MetricType, BuiltInMetric] = None,
                 is_display_plot: bool = True) -> None:
        """
        Default constructor for the Performance metrics taking a dictionary input a
        { metric_name, build in metric}
        This constructor does not automatically include the training and validation losses
          Example {metric='Accuracy', BuildInMetric(MetricType.Accuracy, is_weighted}

        @param registered_perf_metrics: Dictionary { metric_name, built in metric}
        @type registered_perf_metrics: Dictionary
        @param is_display_plot: Flag to display the metric plots at runtime - Default = False
        @type is_display_plot: bool
        """
        self.registered_perf_metrics = registered_perf_metrics if registered_perf_metrics is not None else {}
        self.collected_metrics: Dict[MetricType, List[np.array]] = {}
        self.is_display_plot = is_display_plot

    def __str__(self) -> AnyStr:
        return '\n'.join([f' {k}: {", ".join([str(round(x.item(), 7)) for x in v])}' for k, v in self.collected_metrics.items()])

    @classmethod
    def build(cls, metrics_list: List[MetricType], is_class_imbalance: bool, is_display_plot: bool = False) -> Self:
        """
        Alternative constructor for the Performance metrics taking a dictionary input as
         { metric_name, is_weighted}
        This constructor will automatically add the training and validation losses if they are not explicitly defined
         Example {metric='Accuracy', is_weighted=True}

        @param metrics_list: List of metrics name
        @type metrics_list: Dictionary
        @param is_class_imbalance: Is class imbalance
        @type is_class_imbalance: boolean
        @param is_display_plot: Flag to display the metric plots at runtime - Default = False
        @type is_display_plot: bool
        @return: Instance of PerformanceMetrics
        @rtype: PerformanceMetrics
        """
        assert len(metrics_list) > 0, 'The list of performance metrics is undefined'

        metrics = {metric_type: BuiltInMetric(metric_type=metric_type,
                                              is_weighted=is_class_imbalance) for metric_type in metrics_list}
        return cls(metrics, is_display_plot)

    def reset(self) -> None:
        self.collected_metrics = {}

    def __len__(self) -> int:
        if len(self.collected_metrics) > 0:
            k, v = next(iter(self.collected_metrics.items()))
            return len(v)
        else:
            return 0

    def register_metric(self, metric_type: MetricType, encoding_len: int = -1, is_weighted: bool = False) -> None:
        """
        Add or register a new performance metric such as Precision, Recall, ...
        @param metric_type: Name for the metric (i.e. 'Accuracy')
        @type metric_type: str
        @param encoding_len: Length for the encoding (ie. output Number of classes) with default -1 (no encoding)
        @type encoding_len: int
        @param is_weighted: Specify if the metric is weighted for class distribution imbalance
        @type is_weighted: bool
        """
        self.registered_perf_metrics[metric_type] = BuiltInMetric(metric_type=metric_type,
                                                                  encoding_len=encoding_len,
                                                                  is_weighted=is_weighted)

    def collect_all_metrics(self, epoch: int, metrics_value: Dict[MetricType, torch.Tensor]) -> int:
        """
        Update the current performance metrics for a given epoch and add logging information.
        No assumption is made on the keys of the new set of values.

        @param epoch: Current epoch in the training phase
        @type epoch: int
        @param metrics_value: New slate of metrics for this epoch
        @type metrics_value: Dictionary
        """
        metric_str = '\n'.join([f'   {k.value}: {v.numpy()}' for k, v in metrics_value.items()])
        logging.info(f'>> Epoch: {epoch}\n{metric_str}')
        for key, value in metrics_value.items():
            if key not in self.registered_perf_metrics:
                raise MetricException(f'{key} metric is not supported')

            # Update the existing key values pairs
            if key in self.collected_metrics:
                values = self.collected_metrics[key]
                values.append(value.numpy())
                self.collected_metrics[key] = values
            # In case the new dictionary introduce new key
            else:
                values = [value.numpy()]
                self.collected_metrics[key] = values
        return len(self.collected_metrics.items()) > 0

    def collect_loss(self, is_validation: bool, np_loss: np.array) -> int:
        metric_type = MetricType.EvalLoss if is_validation else MetricType.TrainLoss
        self.collect_metric(metric_type, np_loss)
        return len(self.collected_metrics[metric_type])

    def collect_registered_metrics(self, np_predicted: np.array, np_labeled: np.array) -> None:
        """
        Compute all the registered metrics
        @param np_predicted: Predicted values as Numpy arrays
        @type np_predicted: Numpy array
        @param np_labeled: Labeled values
        @type np_labeled: Numpy array
        """
        for key, metric in self.registered_perf_metrics.items():
            value = metric(np_predicted, np_labeled)
            self.collect_metric(key, value)

    def collect_metric(self, new_metric_type: MetricType, np_new_metric_value: np.array) -> int:
        """
        Update a metric of a given key with a value
        @param new_metric_type: Key or id of the metric
        @type new_metric_type: MetricType
        @param np_new_metric_value: Value of the metric
        @type np_new_metric_value: Numpy array
        """
        if new_metric_type in self.collected_metrics:
            # If this metric is already collected...
            values = self.collected_metrics[new_metric_type]
            values.append(np_new_metric_value)
            self.collected_metrics[new_metric_type] = values
        else:
            # Otherwise create a new entry
            values = [np_new_metric_value]
            self.collected_metrics[new_metric_type] = values
        return len(values)

    def summary(self,
                model_id: AnyStr,
                metric_plotter_parameters: Optional[MetricPlotterParameters] = None) -> None:
        """
        Plots for the various metrics and stored metrics into torch local file which name is derived from
        the model id as
            f'{PerformanceMetrics.perf_metrics_directory}{model_id}.json'

        @param model_id: identifier of the model
        @type model_id str
        @param metric_plotter_parameters: Optional parameters for plotting and storing result of tes
        @type metric_plotter_parameters: MetricPlotterParameters or AnyStr
        """
        if metric_plotter_parameters is not None:
            self.__plot_summary(metric_plotter_parameters)
        self.__save_summary(model_id)

    @staticmethod
    def get_perf_metrics_filename(model_id: AnyStr) -> AnyStr:
        return f'{PerformanceMetrics.perf_metrics_directory}{model_id}.json'

    @staticmethod
    def load_summary(model_id: AnyStr) -> Dict[AnyStr, List[float]]:
        """
        Static method to load the summary of performance metrics as a JSON/Dictionary

        @param model_id: Identifier for the Neural Model as instance of a subclass of NeuralModel
        @type model_id: str
        @return: Dictionary of performance metrics
        @rtype: Dict
        """
        import json

        with open(PerformanceMetrics.get_perf_metrics_filename(model_id), 'r') as f:
            collected_values = json.load(f)
            print(collected_values)
        return collected_values

    """ -------------------------------  Private Helper Methods --------------------------  """

    def __plot_summary(self, metric_plotter_parameters: MetricPlotterParameters) -> None:
        """
        Plots for the various metrics and stored metrics into torch local file

        @param metric_plotter_parameters: Parameter for the plot used to define the file containing the summary of metrics and losses
        @type metric_plotter_parameters: str
        """
        try:
            metric_plotter = MetricPlotter(metric_plotter_parameters)
            # If we need to display the plot in real time
            if self.is_display_plot:
                metric_plotter.plot(self.collected_metrics)
        except FileNotFoundError as e:
            logging.error(f'Output file undefined {e}')
            raise MetricException(e)

    def __save_summary(self, model_id: AnyStr) -> None:
        import json

        collected_values = { k: [float(v) for v in values] for k, values in self.collected_metrics.items()}
        with open(PerformanceMetrics.get_perf_metrics_filename(model_id), 'w') as f:
            json.dump(collected_values, f)



