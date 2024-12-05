__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import AnyStr, Dict, Self, List, Optional
from dl.training.exec_config import ExecConfig
from dl import DLException, TrainingException, ValidationException
from dl.training.hyper_params import HyperParams
from dl.training.early_stop_logger import EarlyStopLogger
from plots.plotter import PlotterParameters
from metric.metric import Metric
from metric.built_in_metric import create_metric_dict
import torch.nn as nn
import logging
logger = logging.getLogger('dl.NeuralNet')

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


class DLTraining(object):
    def __init__(self,
                 hyper_params: HyperParams,
                 early_stop_logger: EarlyStopLogger,
                 metrics: Dict[AnyStr, Metric],
                 exec_config: Optional[ExecConfig] = None,
                 plot_parameters: Optional[List[PlotterParameters]] = None) -> None:
        """
        Constructor for the training and execution of any neural network.
        @param hyper_params: Hyper-parameters associated with the training of th emodel
        @type hyper_params: HyperParams
        @param early_stop_logger: Dynamic condition for early stop in training
        @type early_stop_logger: EarlyStopLogger
        @param metrics: Dictionary of metric {metric_name: build_in_metric instance}
        @type metrics: Dict[AnyStr, Metric]
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @param plot_parameters: Parameters for plotting metrics, training and test losses
        @type plot_parameters: Optional List[PlotterParameters]
        """
        self.hyper_params = hyper_params
        _, self.target_device = exec_config.apply_device()

        self.early_stop_logger = early_stop_logger
        self.plot_parameters = plot_parameters
        self.exec_config = exec_config
        self.metrics: Dict[AnyStr, Metric] = metrics

    @classmethod
    def build(cls,
              hyper_params: HyperParams,
              metric_labels: List[AnyStr]) -> Self:
        """
        Simplified constructor for the training and execution of any neural network.
        @param hyper_params: Hyper parameters associated with the training of th emodel
        @type hyper_params: HyperParams
        @param metric_labels: Labels for metric to be used
        @type metric_labels: List[str]
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        """

        # Create metrics
        metrics_dict = create_metric_dict(metric_labels, hyper_params.encoding_len)
        # Initialize the plotting parameters
        plot_parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                           for label, _ in metrics_dict.items()]

        return cls(hyper_params=hyper_params,
                   early_stop_logger=EarlyStopLogger(patience=2, min_diff_loss=-0.001, early_stopping_enabled=True),
                   metrics=metrics_dict,
                   exec_config=ExecConfig.default(),
                   plot_parameters=plot_parameters)

    @abstractmethod
    def model_label(self) -> AnyStr:
        raise NotImplementedError('NeuralNet.model_label is an abstract method')

    def train(self,
              model_id: AnyStr,
              neural_model: nn.Module,
              train_loader: DataLoader,
              eval_loader: DataLoader) -> None:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and a encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param model_id: Identifier for the model
        @type model_id: str
        @param neural_model: Neural model as torch module
        @type neural_model: nn_Module
        @param train_loader: Data loader for the training set
        @type train_loader: DataLoader
        @param eval_loader:  Data loader for the valuation set
        @type eval_loader: DataLoader
        """
        torch.manual_seed(42)
        output_file_name = f'{model_id}_metrics_{self.plot_parameters[0].title}'
        self.hyper_params.initialize_weight(neural_model.get_modules())

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            train_loss = self.__train(neural_model, epoch, train_loader)

            # Set mode and execute evaluation
            eval_metrics = self.__eval(neural_model, epoch, eval_loader)
            self.early_stop_logger(epoch, train_loss, eval_metrics)
            self.exec_config.apply_monitor_memory()
        # Generate summary
        self.early_stop_logger.summary(output_file_name)
        print(f"\nMPS usage profile for\n{str(self.exec_config)}\n{self.exec_config.accumulator}")

    def __repr__(self) -> str:
        return repr(self.hyper_params)

    """ ------------------------------------   Private methods --------------------------------- """

    def __train(self, neural_model: nn.Module, epoch: int, train_loader: DataLoader) -> float:
        total_loss = 0.0
        # Initialize the gradient for the optimizer
        loss_function = self.hyper_params.loss_function
        optimizer = self.hyper_params.optimizer(neural_model)

        _, torch_device = self.exec_config.apply_device()
        model = neural_model.to(torch_device)
        idx = 0
        for features, labels in train_loader:
            try:
                model.train()

                # Add noise if the mode is defined
                # features = model.add_noise(features)

                # Transfer the input data and labels to the target device
                features = features.to(torch_device)
                labels = labels.to(torch_device)

                predicted = model(features)  # Call forward - prediction
                raw_loss = loss_function(predicted, labels)

                # Set back propagation
                raw_loss.backward(retain_graph=True)
                total_loss += raw_loss.data
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
        return total_loss / len(train_loader)

    def __eval(self, model: nn.Module, epoch: int, eval_loader: DataLoader) -> Dict[AnyStr, float]:
        total_loss = 0
        loss_func = self.hyper_params.loss_function
        metric_collector = {}

        _, torch_device = self.exec_config.apply_device()

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            count = 0
            for features, labels in eval_loader:
                try:
                    model.eval()
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
                    for key, metric in self.metrics.items():
                        value = metric(np_predicted, np_labels)
                        metric_collector[key] = value

                    # Compute and accumulate the loss
                    loss = loss_func(predicted, labels)
                    total_loss += loss.data
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
        metric_collector[Metric.eval_loss_label] = eval_loss
        return metric_collector

