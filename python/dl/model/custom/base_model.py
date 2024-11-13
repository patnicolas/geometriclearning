__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, NoReturn, List, Dict
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dl.block import ConvException
from dl.training.neural_net import NeuralNet
from dl.model.custom.conv_2D_config import Conv2DConfig
from dl.training.hyper_params import HyperParams
from dl.dl_exception import DLException
import logging
logger = logging.getLogger('dl.model.custom.BaseModel')

__all__ = ['BaseModel']


class BaseModel(ABC):

    def __init__(self,
                 conv_2D_config: Conv2DConfig,
                 data_batch_size: int,
                 resize_image: int,
                 subset_size: int = -1) -> None:
        """
          Constructor for any image custom dataset (MNIST, CelebA, ...)
          @param data_batch_size: Size of batch for training
          @type data_batch_size: int
          @param resize_image: Height and width of resized image if > 0, no resize if -1
          @type resize_image: int
          @param subset_size: Subset of data set for training if > 0 the original data set if -1
          @type subset_size: int
          @param conv_2D_config: 2D Convolutional network configuration
          @type conv_2D_config: Conv2DConfig
        """
        self.model = conv_2D_config.conv_model
        self.data_batch_size = data_batch_size
        self.resize_image = resize_image
        self.subset_size = subset_size

    def __repr__(self) -> AnyStr:
        return f'\n{repr(self.model)}\ndata_batch_size {self.data_batch_size }\nResize image: {self.resize_image}' \
                f'\nSubset size: {self.subset_size}'

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

    def load_dataset(self, root_path: AnyStr) -> (DataLoader, DataLoader):
        train_dataset, test_dataset = self._extract_datasets(root_path)

        # If we are experimenting with a subset of the data set for memory usage
        if self.subset_size > 0:
            from torch.utils.data import Subset

            test_subset_size = int(float(self.subset_size * len(test_dataset)) / len(train_dataset))
            train_dataset = Subset(train_dataset, indices=range(self.subset_size))
            test_dataset = Subset(test_dataset, indices=range(test_subset_size))

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.data_batch_size, pin_memory=True, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.data_batch_size, pin_memory=True,shuffle=False)
        return train_loader, test_loader

    """ ---------------------  Private Helper Methods -------------------------- """

    @abstractmethod
    def _extract_datasets(self, root_path: AnyStr) ->(Dataset, Dataset):
        """
        Extract the training data and labels and test data and labels
        @param root_path: Root path to MNIST dataset
        @type root_path: AnyStr
        @return Tuple (train data, test data)
        @rtype Tuple[Dataset]
        """
        raise NotImplementedError('_extract_datasets is an abstract method')




