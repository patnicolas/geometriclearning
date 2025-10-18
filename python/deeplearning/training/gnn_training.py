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


# Standard Library imports
from typing import Dict, AnyStr, Optional, List, Any, Self, Tuple
import logging
# 3rd Party imports
import torch.nn as nn
import torch
import torch_geometric
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# Library imports
from util.monitor_memory_device import monitor_memory_device
from deeplearning.training.neural_training import NeuralTraining
from deeplearning.training.hyper_params import HyperParams
from deeplearning.block.graph import GraphException
from metric.performance_metrics import PerformanceMetrics
from deeplearning.training import TrainingException
from plots.metric_plotter import MetricPlotterParameters
from metric.metric_type import MetricType
from metric.built_in_metric import BuiltInMetric
from deeplearning.training.exec_config import ExecConfig
from deeplearning.training.early_stopping import EarlyStopping
import python
__all__ = ['GNNTraining']


class GNNTraining(NeuralTraining):

    def __init__(self,
                 hyper_params: HyperParams,
                 metrics_attributes: Dict[MetricType, BuiltInMetric],
                 early_stopping: Optional[EarlyStopping] = None,
                 exec_config: ExecConfig = ExecConfig.default(),
                 plot_parameters: Optional[MetricPlotterParameters] = None) -> None:
        """
        Default constructor for this variational auto-encoder
        @param hyper_params:  Hyper-parameters for training and optimization
        @type hyper_params: HyperParams
        @param early_stopping: Early stopping conditions
        @type early_stopping: EarlyStopping
        @param metrics_attributes: Dictionary of metrics and values
        @type metrics_attributes: Dictionary
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @param plot_parameters: Optional plotting parameters
        @type plot_parameters: PlotterParameters
        """
        assert len(metrics_attributes) > 0, 'Metric attributes are undefined'

        if hyper_params.target_device is not None:
            exec_config.device_config = hyper_params.target_device
        super(GNNTraining, self).__init__(hyper_params,
                                          metrics_attributes,
                                          early_stopping,
                                          exec_config,
                                          plot_parameters)

    @classmethod
    def build(cls, training_attributes: Dict[AnyStr, Any], exec_config: ExecConfig=ExecConfig.default()) -> Self:
        # Instantiate the Hyper parameters from the training attributes
        hyper_params = HyperParams.build(training_attributes)

        # Instantiate the Performance metrics
        metric_attributes = PerformanceMetrics.build(metrics_list=training_attributes['metrics_list'],
                                                     is_class_imbalance=training_attributes['is_class_imbalance'])

        # Instantiate the Early Stopping criteria
        early_stopping = EarlyStopping.build(training_attributes)

        # Instantiate the Plots
        train_attrs = training_attributes.get('plot_parameters', None)
        plot_parameters = MetricPlotterParameters.build(train_attrs) if train_attrs is not None else None
        return cls(hyper_params=hyper_params,
                   metrics_attributes=metric_attributes.registered_perf_metrics,
                   early_stopping=early_stopping,
                   exec_config=exec_config,
                   plot_parameters=plot_parameters)

    def __str__(self) -> str:
        metrics_str = '\n'.join([f'   {str(v)}' for _, v in self.performance_metrics.collected_metrics.items()])
        return (f'\nHyper-parameters:\n{repr(self.hyper_params)}'
                f'\nPerformance Metrics\n{metrics_str}'
                f'\nEarly stop condition{self.early_stopping}')

    def get_metric_history(self, metric_type: MetricType) -> List[float]:
        """
        Access the values recorded during training or validation for a given metric of type metric_type
        @param metric_type: Type of the performance metric (i.e. MetricType.Precision)
        @type metric_type: MetricType
        @return: List or history of values recorded for the metric of type metric_type
        @rtype: List[float]
        """
        return self.performance_metrics.performance_values[metric_type]

    def train(self,
              neural_model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader) -> None:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and an encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param neural_model: Neural model as torch module
        @type neural_model: nn_Module
        @param train_loader:  Data loader for the training set
        @type train_loader: torch_geometric.loader.DataLoader
        @param val_loader: Data loader for the evaluation set
        @param val_loader: torch_geometric.loader.DataLoader
        """
        torch.manual_seed(42)

        # Force a conversion to 32-bits
        neural_model = neural_model.float()
        # Reset the parameters for the Graph Neural Layers
        neural_model.reset_parameters()

        logging.info(repr(self.hyper_params))
        logging.info(self.target_device)

        # Train and evaluation process
        for epoch in tqdm(range(self.hyper_params.epochs)):
            # Set training mode and execute training
            self.__train_epoch(neural_model, epoch, train_loader)

            # Set mode and execute evaluation
            if val_loader is not None:
                self.__val_epoch(neural_model, epoch, val_loader)
            logging.info(f'Performance: {epoch=}\n{str(self.performance_metrics)}')
            # self.exec_config.apply_monitor_memory()
        # Generate summary
        self.performance_metrics.summary(neural_model.model_id, self.plot_parameters)
        logging.info(f"\nDevice usage profile for\n{str(self.exec_config)}\n{self.exec_config.accumulator}")

    """ -----------------------------  Private helper methods ------------------------------  """

    def __train_epoch(self, neural_model: nn.Module, epoch: int, train_loader: DataLoader) -> None:
        from python import memory_monitor_enabled

        memory_reports = []
        neural_model.train()
        total_loss = 0.0
        optimizer = self.hyper_params.optimizer(neural_model)

        optimizer.zero_grad(set_to_none=True)
        model = neural_model.to(self.target_device, non_blocking=True)

        for idx, data in enumerate(train_loader):
            try:
                # If memory monitoring is enabled, apply the decorator monitor_memory_device
                if memory_monitor_enabled:
                    loss, report = self.__train_batch(data, model)
                    memory_reports.append(report['end_bytes'])
                else:
                    loss = self.__train_batch(data, model)
                total_loss += loss

                # Empty cache, and batch the accumulation of gradient
                self.exec_config.apply_batch_optimization(idx, optimizer)
            except RuntimeError as e:
                raise GraphException(str(e))
            except AttributeError as e:
                raise GraphException(str(e))
            except ValueError as e:
                raise GraphException(str(e))
        # Collect the training loss
        loss = total_loss/len(train_loader)
        if loss > 1e+6:
            raise TrainingException(f'Loss {loss} for {len(train_loader)} points is out of bounds')

        # Collect the memory statistics for this epoch.
        GNNTraining.__collect_memory(memory_reports)
        # Cleanup
        del memory_reports, model, optimizer
        self.performance_metrics.collect_loss(is_validation=False,
                                              np_loss=np.array(loss))

    @staticmethod
    def __collect_memory(memory_reports: List[AnyStr]) -> None:
        import os
        if not os.path.exists('pin_enabled_3'):
            memory_reports_str = '\n'.join([str(entry) for entry in memory_reports])
            with open('pin_enabled_3', 'wt') as f:
                f.write(memory_reports_str)

    @monitor_memory_device
    def __train_batch(self, data: Tuple[int, Any], model: nn.Module) -> float:
        data = data.to(device=self.target_device, non_blocking=True)
        predicted = model(data)  # Call forward - prediction

        prediction = predicted[data.train_mask]
        expected = data.y[data.train_mask]
        raw_loss = self.hyper_params.loss_function(prediction, expected)

        # Set back propagation
        raw_loss.backward(retain_graph=True)
        return raw_loss.item()

    def __val_epoch(self, model: nn.Module, epoch: int, eval_loader: DataLoader) -> None:
        total_loss = 0
        model.eval()
        predicted_values = []
        labeled_values = []

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            for data in eval_loader:
                try:
                    # Force a conversion to float 32 if necessary
                    if data.x.dtype == torch.float64:
                        data.x = data.x.float()

                    data = data.to(self.target_device)
                    predicted = model(data)  # Call forward - prediction
                    prediction = predicted[data.val_mask]
                    expected = data.y[data.val_mask]
                    raw_loss = self.hyper_params.loss_function(prediction, expected)

                    # Compute and accumulate the loss
                    total_loss += raw_loss.item()

                    # Update the metrics and
                    # Transfer prediction and labels to CPU for computing metrics
                    predicted_values.append(predicted.cpu().numpy())
                    labeled_values.append(data.y.cpu().numpy())
                except RuntimeError as e:
                    raise GraphException(str(e))
                except AttributeError as e:
                    raise GraphException(str(e))
                except ValueError as e:
                    raise GraphException(str(e))

        # Collect all prediction and labels
        all_predicted = np.concatenate(predicted_values)
        all_labeled = np.concatenate(labeled_values)
        del predicted_values, labeled_values

        # Record the values for the registered metrics (e.g., Precision)
        self.performance_metrics.collect_registered_metrics(all_predicted, all_labeled)
        # Record the validation loss
        val_loss = total_loss/len(eval_loader)
        self.performance_metrics.collect_loss(is_validation=True,
                                              np_loss=np.array(val_loss))
