__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from typing import AnyStr, Dict
from dl.model.neural_model import NeuralModel
from dl.training.exec_config import ExecConfig
from dl.dl_exception import DLException
from dl.training.hyper_params import HyperParams
from dl.training.early_stop_logger import EarlyStopLogger
from plots.plotter import PlotterParameters
from metric.metric import Metric
from dl.block import ConvException
from metric.built_in_metric import create_metric_dict
from tqdm import tqdm
from typing import List, Optional, NoReturn
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


class NeuralNet(object):
    def __init__(self,
                 model: NeuralModel,
                 hyper_params: HyperParams,
                 early_stop_logger: EarlyStopLogger,
                 metrics: Dict[AnyStr, Metric],
                 exec_config: ExecConfig,
                 plot_parameters: Optional[List[PlotterParameters]]) -> None:
        """
        Constructor for the training and execution of any neural network.
        @param model: Neural network model (CNN, FeedForward,...)
        @type model: NeuralModel or derived types
        @param hyper_params: Hyper parameters associated with the training of th emodel
        @type hyper_params: HyperParams
        @param early_stop_logger: Dynamic condition for early stop in training
        @type early_stop_logger: EarlyStopLogger
        @param metrics: Dictionary of metric {metric_name: build_in_metric instance}
        @type metrics: Dict[AnyStr, Metric]
        @param plot_parameters: Parameters for plotting metrics, training and test losses
        @type plot_parameters: Optional List[PlotterParameters]
        """
        self.hyper_params = hyper_params
        _, self.target_device = exec_config.apply_device()

        self.model = model.to(self.target_device)
        self.early_stop_logger = early_stop_logger
        self.plot_parameters = plot_parameters
        self.exec_config = exec_config
        self.metrics: Dict[AnyStr, Metric] = metrics

    @classmethod
    def build(cls,
              model: NeuralModel,
              hyper_parameters: HyperParams,
              metric_labels: List[AnyStr],
              exec_config: ExecConfig):
        # Initialize the Early stop mechanism
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)

        # Create metrics
        metrics_dict = create_metric_dict(metric_labels, hyper_parameters.encoding_len)
        # Initialize the plotting parameters
        plot_parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                           for label, _ in metrics_dict.items()]
        return cls(model, hyper_parameters, early_stop_logger, metrics_dict, exec_config, plot_parameters)

    @abstractmethod
    def model_label(self) -> AnyStr:
        raise NotImplementedError('NeuralNet.model_label is an abstract method')

    def train(self,
              train_loader: DataLoader,
              test_loader: DataLoader,
              output_file_name: Optional[AnyStr] = None) -> None:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and a encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param train_loader: Data loader for the training set
        @type train_loader: DataLoader
        @param test_loader:  Data loader for the valuation set
        @type test_loader: DataLoader
        @param output_file_name Optional file name for the output of metrics
        @type output_file_name: AnyStr
        """
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(list(self.model.modules()))

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            train_loss = self.__train(epoch, train_loader)

            # Set mode and execute evaluation
            eval_metrics = self.__eval(epoch, test_loader)
            is_early_stopping = self.early_stop_logger(epoch, train_loss, eval_metrics)
            self.exec_config.apply_monitor_memory()
        # Generate summary
        if self.plot_parameters is not None:
            self.early_stop_logger.summary(output_file_name)

    def execute(self, plot_title: AnyStr, loaders: (DataLoader, DataLoader)) -> None:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param plot_title: Labeling metric for output to file and plots
        @type plot_title: str
        @param loaders: Pair of loader for training data and test data
        @type loaders: Tuple[DataLoader]
        """
        try:
            train_data_loader, test_data_loader = loaders # self.load_dataset(root_path)
            output_file = f'{self.model.model_id}_metrics_{plot_title}'
            self.train(train_data_loader, test_data_loader, output_file)
            # sage_profile = '\n'.join(self.exec_config.accumulator)
            print(f"\nMPS usage profile for\n{str(self.exec_config)}\n{self.exec_config.accumulator}")
        except ConvException as e:
            logger.error(str(e))
            raise DLException(e)
        except AssertionError as e:
            logger.error(str(e))
            raise DLException(e)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                return self.model(features.to(self.target_device))
            except RuntimeError as e:
                raise DLException(str(e))
            except AttributeError as e:
                raise DLException(str(e))
            except Exception as e:
                raise DLException(str(e))

    def init_data_loader(self, batch_size: int, dataset: Dataset) -> (DataLoader, DataLoader):
        torch.manual_seed(42)

        _len = len(dataset)
        train_len = int(_len * self.hyper_params.train_eval_ratio)
        test_len = _len - train_len
        train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
        logger.info(f'Extract {len(train_set)} training and {len(test_set)} test data')

        # Finally initialize the training and test1 loader
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader

    def __repr__(self) -> str:
        return repr(self.hyper_params)

    """ ------------------------------------   Private methods --------------------------------- """

    def __train(self, epoch: int, train_loader: DataLoader) -> float:
        total_loss = 0.0
        # Initialize the gradient for the optimizer
        loss_function = self.hyper_params.loss_function
        optimizer = self.hyper_params.optimizer(self.model)

        _, torch_device = self.exec_config.apply_device()
        model = self.model.to(torch_device)

        for idx, (features, labels) in enumerate(train_loader):
            try:
                model.train()

                features = features.to(torch_device)
                # labels = self.exec_config.apply_labels_dtype(labels)
                labels = labels.to(torch_device)

                # features = self.train_exec_config(features)
                predicted = model(features)  # Call forward - prediction
                raw_loss = loss_function(predicted, labels)
                # print(f'Epoch #{epoch+1}, Batch #{idx+1} Loss: {raw_loss}')

                # Set back propagation
                raw_loss.backward(retain_graph=True)
                total_loss += raw_loss.data
                # Monitoring and caching
                self.exec_config.apply_empty_cache()
                self.exec_config.apply_grad_accu_steps(idx, optimizer)
            except RuntimeError as e:
                raise DLException(str(e))
            except AttributeError as e:
                raise DLException(str(e))
            except ValueError as e:
                raise DLException(f'{str(e)}, features: {str(features)}')
            except Exception as e:
                raise DLException(str(e))
        return total_loss / len(train_loader)

    def __eval(self, epoch: int, test_loader: DataLoader) -> Dict[AnyStr, float]:
        total_loss = 0
        loss_func = self.hyper_params.loss_function
        metric_collector = {}

        _, torch_device = self.exec_config.apply_device()

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            for features, labels in tqdm(test_loader):
                try:
                    self.model.eval()
                    # Transfer the input data to GPU
                    features = features.to(torch_device)
                    labels = labels.to(torch_device)
                    # Execute inference/Prediction
                    predicted = self.model(features)

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
                except RuntimeError as e:
                    raise DLException(str(e))
                except AttributeError as e:
                    raise DLException(str(e))
                except ValueError as e:
                    raise DLException(str(e))
                except Exception as e:
                    raise DLException(str(e))

        eval_loss = total_loss / len(test_loader)
        metric_collector[Metric.eval_loss_label] = eval_loss
        return metric_collector

