__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn
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
from metric.built_in_metric import BuiltInMetric, MetricType
import logging
logger = logging.getLogger('dl.model.custom.BaseMNIST')

__all__ = ['BaseMnist']


class BaseMnist(ABC):
    default_training_file = 'processed/training.pt'
    default_test_file = 'processed/test.pt'
    num_classes = 10

    def __init__(self, model: NeuralModel) -> None:
        self.model = model

    def __repr__(self) -> AnyStr:
        return repr(self.model)

    def do_train(self,
                 root_path: AnyStr,
                 hyper_parameters: HyperParams,
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

            train_data_loader, test_data_loader = self.load_dataset(root_path, use_labels=True)
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

    def load_dataset(self, root_path: AnyStr, use_labels: bool) -> (DataLoader, DataLoader):
        import torch
        is_testing = False

        if is_testing:
            print('Random data')
            train_features = torch.randn(640, 1, 28, 28)
            train_labels = torch.randn(640)
            test_features = torch.randn(64, 1, 28, 28)
            test_labels = torch.randn(64)
        else:
            train_features, train_labels, test_features, test_labels = self._extract_datasets(root_path)

        # Build the data set as PyTorch tensors
        if use_labels:
            train_dataset = TensorDataset(train_features, train_labels)
            test_dataset = TensorDataset(test_features, test_labels)
        else:
            train_dataset = TensorDataset(train_features)[0]
            test_dataset = TensorDataset(test_features)[0]

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader



