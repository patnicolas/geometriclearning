__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn
from torch.utils.data import DataLoader, TensorDataset
from dl.training.neuralnet import NeuralNet
from dl.training.earlystoplogger import EarlyStopLogger
from dl.model.neuralmodel import NeuralModel
from metric.metric import Metric
from plots.plotter import PlotterParameters
from dl.training.hyperparams import HyperParams
from metric.builtinmetric import BuiltInMetric, MetricType
import logging
logger = logging.getLogger('dl.model.custom.BaseMNIST')




class BaseMNIST(object):
    default_training_file = 'processed/training.pt'
    default_test_file = 'processed/test.pt'
    num_classes = 10

    def __init__(self, model: NeuralModel) -> None:
        self.model = model

    def do_train(self, root_path: AnyStr, hyper_parameters: HyperParams) -> NoReturn:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param root_path: Path for the root of the MNIST data
        @type root_path: str
        @param hyper_parameters: Hyper-parameteres for the execution of the
        @type hyper_parameters: HyperParams
        """
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

        # Define the neural network as model, hyper-parameters, early stopping criteria and metrics
        network = NeuralNet(
            self.model,
            hyper_parameters,
            early_stop_logger,
            metric_labels,
            parameters)

        train_data_loader, test_data_loader = BaseMNIST.__load_dataset(root_path)
        network(train_data_loader, test_data_loader)

    """ ---------------------  Private Helper Methods -------------------------- """

    @staticmethod
    def __load_dataset(root_path: AnyStr) -> (DataLoader, DataLoader):
        import torch
        is_testing = False

        if is_testing:
            print('Random data')
            train_features = torch.randn(640, 1, 28, 28)
            train_labels = torch.randn(640)
            test_features = torch.randn(64, 1, 28, 28)
            test_labels = torch.randn(64)
        else:
            target_device, torch_device = NeuralNet.get_device()
            print(f'Real data for device {target_device}')

            train_data = torch.load(f'{root_path}/{BaseMNIST.default_training_file}')
            train_features = train_data[0].unsqueeze(dim=1).float().to(torch_device)
            train_labels = torch.nn.functional.one_hot(train_data[1], num_classes=10).float().to(torch_device)

            test_data = torch.load(f'{root_path}/{BaseMNIST.default_test_file}')
            test_features = test_data[0].unsqueeze(dim=1).float().to(torch_device)
            test_labels = torch.nn.functional.one_hot(test_data[1], num_classes=10).float().to(torch_device)

        # Build the data set as PyTorch tensors
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader



