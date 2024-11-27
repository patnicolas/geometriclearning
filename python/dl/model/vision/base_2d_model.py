__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, List
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from dl import ConvException
from dl.training.neural_net_training import NeuralNetTraining
from dl.model.vision.conv_2d_config import Conv2DConfig
from dl.training.hyper_params import HyperParams
from dl import DLException
from dl.training.exec_config import ExecConfig
import logging
logger = logging.getLogger('dl.model.vision.BaseModel')

__all__ = ['Base2DModel']


class Base2DModel(ABC):

    def __init__(self, conv_2D_config: Conv2DConfig, data_batch_size: int, resize_image: int) -> None:
        """
          Constructor for any image vision dataset (MNIST, CelebA, ...)
          @param data_batch_size: Size of batch for training
          @type data_batch_size: int
          @param resize_image: Height and width of resized image if > 0, no resize if -1
          @type resize_image: int
          @param conv_2D_config: 2D Convolutional network configuration
          @type conv_2D_config: Conv2DConfig
        """
        self.model = conv_2D_config.conv_model
        self.data_batch_size = data_batch_size
        self.resize_image = resize_image

    def __repr__(self) -> AnyStr:
        return f'\n{repr(self.model)}\ndata_batch_size {self.data_batch_size }\nResize image: {self.resize_image}'

    def do_train(self,
                 root_path: AnyStr,
                 hyper_parameters: HyperParams,
                 metric_labels: List[AnyStr],
                 exec_config: ExecConfig,
                 plot_title: AnyStr) -> None:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param root_path: Path for the root of the MNIST data
        @type root_path: str
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metric_labels: List of metrics to be used
        @type metric_labels: List
        @param exec_config: Configuration for the execution of training set
        @type exec_config: ExecConfig
        @param plot_title: Labeling metric for output to file and plots
        @type plot_title: str
        """
        try:
            network = NeuralNetTraining.build(self.model, hyper_parameters, metric_labels, exec_config)
            plot_title = f'{self.model.model_id}_metrics_{plot_title}'
            network(plot_title=plot_title, loaders=self.load_dataset(root_path, exec_config))
        except ConvException as e:
            logger.error(str(e))
            raise DLException(e)
        except AssertionError as e:
            logger.error(str(e))
            raise DLException(e)

    def load_dataset(self, root_path: AnyStr, exec_config: ExecConfig) -> (DataLoader, DataLoader):
        train_dataset, test_dataset = self._extract_datasets(root_path)

        # If we are experimenting with a subset of the data set for memory usage
        train_dataset, test_dataset = exec_config.apply_sampling(train_dataset,  test_dataset)

        # Create DataLoaders for batch processing
        train_loader, test_loader = exec_config.apply_optimize_loaders(self.data_batch_size, train_dataset, test_dataset)
        return train_loader, test_loader

    """ ---------------------  Private Helper Methods -------------------------- """

    @abstractmethod
    def _extract_datasets(self, root_path: AnyStr) ->(Dataset, Dataset):
        """
        Extract the training data and labels and test data and labels. This method has to be overwritten
        in subclasses such as convolutional model for MNIST data set...
        @param root_path: Root path to MNIST dataset
        @type root_path: AnyStr
        @return Tuple (train dataset, test dataset)
        @rtype Tuple[Dataset]
        """
        raise NotImplementedError('_extract_datasets is an abstract method')




