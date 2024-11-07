__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Tuple, NoReturn
import torch
from dl.model.custom.base_mnist import BaseMnist
from dl.model.custom.conv_2D_config import Conv2DConfig
import logging
logger = logging.getLogger('dl.model.custom.ConvMNIST')

__all__ = ['ConvMNIST']


class ConvMNIST(BaseMnist):
    id = 'Convolutional_MNIST'

    def __init__(self, conv_2D_config: Conv2DConfig, data_batch_size: int = 64) -> None:
        """
        Constructor for the Convolutional network for MNIST Dataset
        @param conv_2D_config: 2D Convolutional network
        @type conv_2D_config: Conv2DConfig
        """
        super(ConvMNIST, self).__init__(conv_2D_config.conv_model, data_batch_size)

    def show_conv_weights_shape(self) -> NoReturn:
        import torch

        for idx, conv_block in enumerate(self.model.conv_blocks):
            conv_modules_weights: Tuple[torch.Tensor] = conv_block.get_modules_weights()
            logging.info(f'\nConv. layer #{idx} shape: {conv_modules_weights[0].shape}')

    def _extract_datasets(self, root_path: AnyStr) ->(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to MNIST dataset
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        from dl.training.neural_net import NeuralNet
        from torch.nn.functional import one_hot

        _, torch_device = NeuralNet.get_device()

        train_data = torch.load(f'{root_path}/{BaseMnist.default_training_file}')
        train_features = train_data[0].unsqueeze(dim=1).float().to(torch_device)
        train_labels = one_hot(train_data[1], num_classes=BaseMnist.num_classes).float().to(torch_device)

        test_data = torch.load(f'{root_path}/{BaseMnist.default_test_file}')
        test_features = test_data[0].unsqueeze(dim=1).float().to(torch_device)
        test_labels = one_hot(test_data[1], num_classes=BaseMnist.num_classes).float().to(torch_device)

        return train_features, train_labels, test_features, test_labels
