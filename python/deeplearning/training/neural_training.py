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
from typing import AnyStr, Dict, List, Optional
import logging
# 3rd Party imports
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
# Library imports
from deeplearning.training.exec_config import ExecConfig
from deeplearning.training import TrainingException, ValidationException
from deeplearning.training.hyper_params import HyperParams
from metric.built_in_metric import BuiltInMetric
from metric.metric import Metric
from metric.metric_type import MetricType
from plots.metric_plotter import MetricPlotterParameters
from metric.performance_metrics import PerformanceMetrics
from deeplearning.training.early_stopping import EarlyStopping
import python
__all__ = ['NeuralTraining']


class NeuralTraining(object):
    """
        Generic Neural Network abstract class. There are 2 version of train and evaluation
        - _train_and_evaluate Training and evaluation from a pre-configure train loader
        -  train_and_evaluate Training and evaluation from a raw data set
        The method transform_label has to be overwritten in the inheriting classes to support
        transformation/conversion of labels if needed.
        The following methods have to be overwritten in derived classes
        - transform_label Transform the label input_tensor if necessary
        - model_label Model identification
    """
    def __init__(self,
                 hyper_params: HyperParams,
                 metrics_attributes: Dict[MetricType, BuiltInMetric],
                 early_stopping: Optional[EarlyStopping] = None,
                 exec_config: Optional[ExecConfig] = None,
                 plot_parameters: Optional[MetricPlotterParameters] = None) -> None:
        """
        Constructor for the training and execution of any neural network.
        @param hyper_params: Hyper-parameters associated with the training of th emodel
        @type hyper_params: HyperParams
        @param metrics_attributes: Dictionary of metric {metric_name: build_in_metric instance}
        @type metrics_attributes: Dict[AnyStr, Metric]
        @param early_stopping: Optional early stopping conditions
        @type early_stopping: EarlyStopping
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @param plot_parameters: Parameters for plotting metrics, training and test losses
        @type plot_parameters: Optional List[PlotterParameters]
        """
        assert len(metrics_attributes) > 0, 'Metric attributes are undefined'

        self.hyper_params = hyper_params
        _, self.target_device = exec_config.apply_device()
        self.early_stopping = early_stopping
        self.plot_parameters = plot_parameters
        self.exec_config = exec_config
        self.performance_metrics = PerformanceMetrics(metrics_attributes)

    def reset_perf_metrics(self) -> None:
        self.performance_metrics.reset()

    def train(self,
              neural_model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader) -> None:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and a encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param neural_model: Neural model as torch module
        @type neural_model: nn_Module
        @param train_loader: Data loader for the training set
        @type train_loader: DataLoader
        @param val_loader:  Data loader for the valuation set
        @type val_loader: DataLoader
        """
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(neural_model.get_modules())
        neural_model.reset_parameters()

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            self.__train_epoch(neural_model, epoch, train_loader)

            # Set mode and execute evaluation
            if val_loader is not None:
                self.__val_epoch(neural_model, epoch, val_loader)
            self.exec_config.apply_monitor_memory()

        # Generate summary
        self.performance_metrics.summary(self.plot_parameters.plot_filename)
        logging.info(f"\nMPS usage profile for\n{str(self.exec_config)}\n{self.exec_config.accumulator}")

    def __repr__(self) -> str:
        return repr(self.hyper_params)

    """ ------------------------------------   Private methods --------------------------------- """

    def __train_epoch(self, neural_model: nn.Module, epoch: int, train_loader: DataLoader) -> None:
        total_loss = 0.0

        # Initialize the gradient for the optimizer
        loss_function = self.hyper_params.loss_function
        optimizer = self.hyper_params.optimizer(neural_model)

        _, torch_device = self.exec_config.apply_device()
        model = neural_model.to(torch_device, non_blocking=True)
        model.train()
        idx = 0

        for features, labels in train_loader:
            try:
                # Add noise if the mode is defined
                # features = model.add_noise(features)
                # Transfer the input data and labels to the target device
                features = features.to(device=torch_device, non_blocking=True)
                labels = labels.to(device=torch_device, non_blocking=True)

                predicted = model(features)  # Call forward - prediction
                raw_loss = loss_function(predicted, labels)

                # Set back propagation
                raw_loss.backward(retain_graph=True)
                total_loss += raw_loss.item()

                # Monitoring and caching for performance imp
                self.exec_config.apply_empty_cache()
                self.exec_config.apply_grad_accu_steps(idx, optimizer)
                idx += 1
            except RuntimeError as e:
                raise TrainingException(str(e))
            except AttributeError as e:
                raise TrainingException(str(e))
            except ValueError as e:
                raise TrainingException(f'{str(e)}, features: {str(features)}')
            except Exception as e:
                raise TrainingException(str(e))
            average_loss = total_loss / len(train_loader)
            self.performance_metrics.collect_metric(MetricType.TrainLoss, np.array(average_loss))

    def __val_epoch(self, model: nn.Module, epoch: int, eval_loader: DataLoader) -> None:
        total_loss = 0
        model.eval()
        loss_func = self.hyper_params.loss_function

        _, torch_device = self.exec_config.apply_device()

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            count = 0
            for features, labels in eval_loader:
                try:
                    # Add noise if the mode is defined
                    # features = model.add_noise(features)

                    # Transfer the input data to GPU
                    features = features.to(torch_device)
                    labels = labels.to(torch_device)
                    # Execute inference/Prediction
                    predicted = model(features)

                    # Transfer prediction and labels to CPU for metrics
                    np_predicted = predicted.cpu().numpy()
                    np_labels = labels.cpu().numpy()

                    # Update the metrics
                    self.performance_metrics.update_performance_values(np_predicted, np_labels)

                    # Compute and accumulate the loss
                    total_loss += loss_func(predicted, labels)
                    count += 1
                except RuntimeError as e:
                    raise ValidationException(str(e))
                except AttributeError as e:
                    raise ValidationException(str(e))
                except ValueError as e:
                    raise ValidationException(str(e))
                except Exception as e:
                    raise ValidationException(str(e))

        eval_loss = total_loss / count
        self.performance_metrics.collect_metric(MetricType.EvalLoss, np.array(eval_loss))

