__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dl.model.vision.conv_2D_config import Conv2DConfig
from typing import AnyStr
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from dl.model.vision.base_model import BaseModel
from dl.model.vision import GrayscaleToRGB
import logging
logger = logging.getLogger('dl.model.vision.ConvCaltech101')
logging.basicConfig(level=logging.INFO)


class ConvCaltech101(BaseModel):
    id = 'Convolutional_Caltech101'

    def __init__(self,
                 conv_2D_config: Conv2DConfig,
                 data_batch_size: int,
                 resize_image: int,
                 train_test_split: float = 0.85) -> None:
        """
            Constructor for any image vision dataset (MNIST, CelebA, ...)
            @param data_batch_size: Size of batch for training
            @type data_batch_size: int
            @param resize_image: Height and width of resized image if > 0, no resize if -1
            @type resize_image: int
            @param train_test_split: Specify the ratio size training set / size testing ste
            @type train_test_split: float
            @param conv_2D_config: 2D Convolutional network configuration
            @type conv_2D_config: Conv2DConfig
        """
        assert 0.5 < train_test_split < 0.98, \
            f'Ratio training set / test set {train_test_split} is out of bounds ]0.5, 0.98['

        super(ConvCaltech101, self).__init__(conv_2D_config, data_batch_size, resize_image)
        self.train_test_split = train_test_split

    def _extract_datasets(self, root_path: AnyStr) -> (Dataset, Dataset):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to CIFAR10 data
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        import torch
        from torchvision import transforms
        from torch.utils.data import DataLoader
        from torchvision.datasets.caltech import Caltech101

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize(size=(self.resize_image, self.resize_image), interpolation=InterpolationMode.BILINEAR),
            GrayscaleToRGB(),
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5))
        ]) if self.resize_image > 0 else transforms.Compose([
            GrayscaleToRGB(),
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5))
        ])

        caltech_101_dataset = Caltech101(root=root_path, transform=transform, download=False)
        train_size = int(self.train_test_split * len(caltech_101_dataset))
        test_size = len(caltech_101_dataset) - train_size
        return torch.utils.data.random_split(caltech_101_dataset, lengths=[train_size, test_size])
