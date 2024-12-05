__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dataset import DatasetException
from dataset.base_loader import BaseLoader
from typing import AnyStr
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from dl.model.vision import GrayscaleToRGB
import logging
logger = logging.getLogger('dataset.Caltech101Loader')


class Caltech101Loader(BaseLoader):
    plot_layout = (4, 4)

    def __init__(self, split_ratio: float, resize_image: int = -1) -> None:
        """
        Constructor for the Caltech-101 data set
        @param split_ratio: Training-validation random split ratio
        @type split_ratio: float
        @param resize_image: Size of image to be resized. The image is not resized if -1
        @type resize_image: int
        """
        assert 0.5 <= split_ratio <= 0.95, f'Training-validation split ratio {split_ratio} should be [0.5, 0.95]'

        super(Caltech101Loader, self).__init__(num_samples=-1)
        self.split_ratio = split_ratio
        self.resize_image = resize_image

    @staticmethod
    def show_samples(data_path: AnyStr, is_random: bool = True) -> None:
        """
        Show a random sample of images from a given dataset
        @param data_path: Path for the Caltech 101 images
        @type data_path: str
        @param is_random: Flag to specify the images have to be randomly selected
        @type is_random: bool
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        import os

        category_path = f'{data_path}/101_ObjectCategories'
        images, labels = Caltech101Loader.__extract_images_and_labels(category_path)
        fig, axes = plt.subplots(nrows=Caltech101Loader.plot_layout[0],
                                 ncols=Caltech101Loader.plot_layout[1],
                                 figsize=(12, 6))

        for idx, ax in enumerate(axes.flat):
            img_path = os.path.join(category_path, images[idx])
            img = Image.open(img_path)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            title = labels[idx]
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

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
        from torchvision.datasets.caltech import Caltech101

        try:
            # Define image transformations with or without resizing
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

            # Instantiate the data set
            caltech_101_dataset = Caltech101(root=root_path, transform=transform, download=False)

            # Split training / validation data sets.
            train_size = int(self.split_ratio * len(caltech_101_dataset))
            test_size = len(caltech_101_dataset) - train_size
            return torch.utils.data.random_split(caltech_101_dataset, lengths=[train_size, test_size])

        except RuntimeError as e:
            logger.error(str(e))
            raise DatasetException(str(e))

    @staticmethod
    def __extract_images_and_labels(category_path: AnyStr, is_random: bool) -> (List[AnyStr], List[AnyStr]):
        import os
        import random

        num_plots = Caltech101Loader.plot_layout[0] * Caltech101Loader.plot_layout[1]
        #category_path = f'{data_path}/101_ObjectCategories'
        category_dirs = os.listdir(category_path)
        categories_indices = [random.randint(a=0, b=len(category_dirs) - 1) if is_random else 0 for _ in range(num_plots)]
        images = [f'{category_dirs[category_index]}/image_0001.jpg' for category_index in categories_indices]
        labels = [category_dirs[category_index] for category_index in categories_indices]

        return images, labels

    @staticmethod
    def __extract_categoryX(category_path: AnyStr) -> AnyStr:
        all_categories_dir = '101_ObjectCategories'
        end_index = category_path.find('/image')
        start_index = category_path.find(all_categories_dir)
        category = category_path[start_index + len(all_categories_dir): end_index]
        return category
