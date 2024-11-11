__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dl.model.custom.conv_2D_config import Conv2DConfig
from typing import AnyStr, NoReturn, List
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader
from dl.training.neural_net import NeuralNet
from dl.dl_exception import DLException
from dl.block import ConvException
from dl.training.hyper_params import HyperParams
import logging
logger = logging.getLogger('dl.model.custom.ConvCifar10')


class ConvCelebA(object):
    id = 'Convolutional_CelebA'

    def __init__(self, conv_2D_config: Conv2DConfig, data_batch_size: int = 32) -> None:
        self.model = conv_2D_config.conv_model
        self.data_batch_size = data_batch_size

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
        # Create the training and evaluation data sets
        train_dataset, test_dataset = ConvCelebA.__extract_datasets(root_path)

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.data_batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.data_batch_size, shuffle=False)
        return train_loader, test_loader

    def __repr__(self) -> AnyStr:
        return repr(self.model)

    """ ---------------------------  Private helper methods ---------------------------- """

    @staticmethod
    def __extract_datasets(root_path: AnyStr) -> (CIFAR10, CIFAR10):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to CIFAR10 data
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        from dl.training.neural_net import NeuralNet

        # Extract the processing device (CPU, Cuda,...)
        _, torch_device = NeuralNet.get_device()

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            # Normalize with mean and std for RGB channels
            transforms.Normalize(mean =(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_dataset = CelebA(
            root=root_path,  # Directory to store the dataset
            split='train',  # Load training data
            download=True,  # Download if not already present
            transform=transform  # Apply transformations
        )

        test_dataset = CelebA(
            root=root_path,  # Directory to store the dataset
            split='test',  # Load test data
            download=True,  # Download if not already present
            transform=transform  # Apply transformations
        )
        return train_dataset, test_dataset





