__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Tuple, NoReturn
import torch
from dl.model.vision.base_model import BaseModel
from dl.model.vision.conv_2D_config import Conv2DConfig
from torch.utils.data import  TensorDataset, Dataset
import logging
logger = logging.getLogger('dl.model.vision.ConvMNIST')

__all__ = ['ConvMNIST']


class ConvMNIST(BaseModel):
    id = 'Convolutional_MNIST'

    default_training_file = 'processed/training.pt'
    default_test_file = 'processed/test.pt'
    num_classes = 10

    def __init__(self,
                 conv_2D_config: Conv2DConfig,
                 data_batch_size: int,
                 resize_image: int,
                 subset_size: int = -1) -> None:
        """
        Constructor for the Convolutional network for MNIST Dataset
        @param data_batch_size: Size of batch for training
        @type data_batch_size: int
        @param resize_image: Height and width of resized image if > 0, no resize if -1
        @type resize_image: int
        @param subset_size: Subset of data set for training if > 0 the original data set if -1
        @type subset_size: int
        @param conv_2D_config: 2D Convolutional network configuration
        @type conv_2D_config: Conv2DConfig
        """
        super(ConvMNIST, self).__init__(conv_2D_config, data_batch_size, resize_image, subset_size)

    def show_conv_weights_shape(self) -> NoReturn:
        import torch

        for idx, conv_block in enumerate(self.model.conv_blocks):
            conv_modules_weights: Tuple[torch.Tensor] = conv_block.get_modules_weights()
            logging.info(f'\nConv. layer #{idx} shape: {conv_modules_weights[0].shape}')

    def _extract_datasets(self, root_path: AnyStr) ->(Dataset, Dataset):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to MNIST dataset
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        from torch.nn.functional import one_hot

        train_data = torch.load(f'{root_path}/{ConvMNIST.default_training_file}')
        train_features = train_data[0].unsqueeze(dim=1).float()
        train_labels = one_hot(train_data[1], num_classes=ConvMNIST.num_classes).float()

        test_data = torch.load(f'{root_path}/{ConvMNIST.default_test_file}')
        test_features = test_data[0].unsqueeze(dim=1).float()
        test_labels = one_hot(test_data[1], num_classes=ConvMNIST.num_classes).float()

        train_dataset: Dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        return train_dataset, test_dataset
