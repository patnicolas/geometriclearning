__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dl.model.custom.conv_2D_config import Conv2DConfig
from typing import AnyStr, NoReturn, List
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from dl.training.neural_net import NeuralNet
from dl.dl_exception import DLException
from dl.block import ConvException
from dl.training.hyper_params import HyperParams
import logging
logger = logging.getLogger('dl.model.custom.ConvCaltech101')
logging.basicConfig(level=logging.INFO)


class ConvCaltech101(object):
    id = 'Convolutional_Caltech101'

    def __init__(self,
                 conv_2D_config: Conv2DConfig,
                 data_batch_size: int,
                 resize_image: int,
                 subset_size: int =-1) -> None:
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
        super(ConvCaltech101, self).__init__(conv_2D_config, data_batch_size, resize_image, subset_size)

