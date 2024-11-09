__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, List, Dict
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, TensorDataset
from dl.block import ConvException
from dl.training.neural_net import NeuralNet
from dl.model.neural_model import NeuralModel
from dl.training.hyper_params import HyperParams
from dl.dl_exception import DLException
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
                 plot_title: AnyStr) -> NoReturn:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param root_path: Path for the root of the MNIST data
        @type root_path: str
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metric_labels: List of metrics to be used
        @type metric_labels: List
        @param plot_title: Labeling metric for output to file and plots
        @type plot_title: str
        """
        try:
            network = NeuralNet.build(self.model, hyper_parameters, metric_labels)
            plot_title = f'{self.model.model_id}_metrics_{plot_title}'
            network.execute(plot_title=plot_title, loaders=self.load_dataset(root_path))
        except ConvException as e:
            logger.error(str(e))
            raise DLException(e)
        except AssertionError as e:
            logger.error(str(e))
            raise DLException(e)

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
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.data_batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.data_batch_size, shuffle=False)
        return train_loader, test_loader



