__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, List, Dict
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl.block import ConvException
from dl.training.neural_net import NeuralNet
from dl.training.early_stop_logger import EarlyStopLogger
from dl.model.neural_model import NeuralModel
from metric.metric import Metric
from plots.plotter import PlotterParameters
from dl.training.hyper_params import HyperParams
from dl.dl_exception import DLException
from metric.built_in_metric import BuiltInMetric, MetricType, create_metric_dict
import logging
logger = logging.getLogger('dl.model.custom.BaseMNIST')

__all__ = ['BaseMnist']


class BaseMnist(ABC):
    default_training_file = 'processed/training.pt'
    default_test_file = 'processed/test.pt'
    num_classes = 10

    def __init__(self, model: NeuralModel, data_batch_size: int = 64) -> None:
        self.model = model
        self.data_batch_size = data_batch_size

    def __repr__(self) -> AnyStr:
        return repr(self.model)

    def do_train(self,
                 root_path: AnyStr,
                 hyper_parameters: HyperParams,
                 metric_labels: List[AnyStr],
                 metric_label: AnyStr) -> NoReturn:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param root_path: Path for the root of the MNIST data
        @type root_path: str
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metric_label: Labeling metric for output to file and plots
        @type metric_label: str
        """
        try:
            patience = 2
            min_diff_loss = -0.001
            early_stopping_enabled = True
            early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
            metric_labels = create_metric_dict(metric_labels)
            parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                          for label, _ in metric_labels.items()]

            # Define the neural network as model, hyperparameters, early stopping criteria and metrics
            network = NeuralNet(
                self.model,
                hyper_parameters,
                early_stop_logger,
                metric_labels,
                parameters)

            train_data_loader, test_data_loader = self.load_dataset(root_path)
            output_file = f'{self.model.model_id}_metrics_{metric_label}'
            network(train_data_loader, test_data_loader, output_file)
        except ConvException as e:
            logging.error(str(e))
        except DLException as e:
            logging.error(str(e))

    def do_train2(self,
                 root_path: AnyStr,
                 hyper_parameters: HyperParams,
                 metric_labels: List[AnyStr],
                 metric_label: AnyStr) -> NoReturn:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param root_path: Path for the root of the MNIST data
        @type root_path: str
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metric_label: Labeling metric for output to file and plots
        @type metric_label: str
        """
        try:
            patience = 2
            min_diff_loss = -0.001
            early_stopping_enabled = True
            early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
            metric_labels = {
                Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, is_weighted=True),
                Metric.precision_label: BuiltInMetric(MetricType.Precision, is_weighted=True)
            }
            parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                          for label, _ in metric_labels.items()]

            # Define the neural network as model, hyperparameters, early stopping criteria and metrics
            network = NeuralNet(
                self.model,
                hyper_parameters,
                early_stop_logger,
                metric_labels,
                parameters)

            train_data_loader, test_data_loader = self.load_dataset(root_path)
            output_file = f'{self.model.model_id}_metrics_{metric_label}'
            network(train_data_loader, test_data_loader, output_file)
        except ConvException as e:
            logging.error(str(e))
        except DLException as e:
            logging.error(str(e))

    @abstractmethod
    def _extract_datasets(self, root_path: AnyStr) ->(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Extract the training data and labels and test data and labels
        @param root_path: Root path to MNIST dataset
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        raise NotImplementedError('NeuralNet.model_label is an abstract method')

    """ ---------------------  Private Helper Methods -------------------------- """

    def load_dataset(self, root_path: AnyStr) -> (DataLoader, DataLoader):
        train_features, train_labels, test_features, test_labels = self._extract_datasets(root_path)

        # Build the data set as PyTorch tensors
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader



