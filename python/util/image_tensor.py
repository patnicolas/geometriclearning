__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 3rd Party imports
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
__all__ = ['ImageTensor']
default_output_dir = '../output/images'




class ImageTensor(object):
    """
        Conversion between a input_tensor into an image
        to_images Generate images from a list of input_tensors
        to_image Generate an image from a single input_tensor
        to_dataset Generate a data loader from a list of images

    """
    def __init__(self, images_dir: str, img_scale_factor: int = None) -> None:
        """
        Constructor for the conversion of a tensor into an image

        @param images_dir: Directory containing input_tensor images
        @param img_scale_factor: Scale factor for this image, If scale factor is undefined (None) images are not displayed
        during processing
        """
        (ImageTensor, self).__init__()
        self.images_dir = images_dir
        self.img_scale_factor = img_scale_factor

    @staticmethod
    def to_dataset() -> DataLoader:
        dataset = datasets.ImageFolder(default_output_dir, transform=transforms.ToTensor())
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return data_loader

    def to_images(self, input_tensors: torch.Tensor, n_channels: int = 3):
        """
            Convert an aggregated tensor, convert it to a list of images to be potentially displayed and saved
            @param input_tensors: Tensor as an aggregated of multiple tensors, each representing an image
            @param n_channels: Number of channels (1 for grey, 3 for RGB
        """
        shapes = list(input_tensors.size())
        num_images = 16 if shapes[0] > 16 else shapes[0]

        imgs = [self.__to_image(input_tensors[idx], str(idx), n_channels) for idx in range(num_images)]
        if self.img_scale_factor is not None:
            ImageTensor.__show_images(imgs, shapes)


    def to_image(self, input_tensor: torch.Tensor, img_name: str, n_channels: int = 3):
        """
            Convert a 3 dimension input into an image. The dimension (shapes) are defined as
            1 Width of the image
            2 Height of the image
            3 Number of channels (1 for shades of greay, 3 for RGB)
            @param input_tensor: Input input
            @param file_name: Name of the images
            @param n_channels: Number of channels should be either 1 or 3
        """
        img = ImageTensor.__to_image(input_tensor, img_name, n_channels)
        # Display and save if necessary
        if self.img_scale_factor is not None:
            self.__show_image(img, img_name)
        img.save(f'{self.images_dir}/{img_name}.png')
        del img

    def __show_image(self, img: Image, img_name: str):
        resize_h = img.size[0] * self.img_scale_factor
        resize_b = img.size[1] * self.img_scale_factor
        img.resize((resize_h, resize_b)).show(img_name)


    @staticmethod
    def __to_image(input: torch.Tensor, img_name: str, n_channels: int = 3):
        """
            Convert a 3 dimension input_tensor into an image. The dimension (shapes) are defined as
            1 Width of the image
            2 Height of the image
            3 Number of channels (1 for shades of greay, 3 for RGB)
            @param input: Input input_tensor
            @param file_name: Name of the images
            @param n_channels: Number of channels should be either 1 or 3
            @param to_show: Specify that the image has to be shown.
        """
        from PIL import Image

        assert n_channels in [1, 3], f'Number of channels {n_channels} should be {1, 3}'
        shapes = list(input.size())
        assert len(shapes) == n_channels, f'ImageTensor.to_image: Num shapes {len(shapes)} should be 3'
        assert shapes[2] == n_channels, f'ImageTensor.to_image: 3rd dimension {shapes[2]} should be {n_channels}'

        t = torch.full((shapes[0], shapes[1], shapes[2]), fill_value = 0, dtype=torch.uint8)

        def generator(i: int, j: int, k: int, cell: float):
            t[i, j, k] = (cell * 255).int()

        [generator(i, j, k, input[i, j, k]) for i in range(shapes[0])
         for j in range(shapes[1]) for k in range(shapes[2]) if input[i, j, k] > 0.0]
        data = t.numpy()
        return Image.fromarray(data)


    @staticmethod
    def show_image(input_data: list, target_data: list, title: str, num_items: int):
        """
            Display image input_tensors for debugging purpose. The total number of images displayed are num_cols by 6 rows
            @param input_data: List of torch input_tensor for images
            :type input_data: lst
            @param target_data: List of torch input_tensor for labels
            :type target_data: lst
            @param num_items: Number of items to display
            :type num_items: int
            @param title: Title of the graph
            :type title: str
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for i in range(num_items):
            plt.subplot(3, num_items / 2, i + 1)
            plt.tight_layout()
            plt.imshow(input_data[i][0], cmap='gray', interpolation='none')
            plt.title(f'{title}: {target_data[i]}')
            plt.xticks([])
            plt.yticks([])
        fig.show()


    @staticmethod
    def show_tensor_images(image_tensor: torch.Tensor, num_images: int, num_row: int):
        """
            Function for visualizing images: Given a input_tensor of images, number of images, and
            size per image, plots and prints the images in an uniform grid.
            @param image_tensor: Tensor representing num_images
            @param num_images: Number of images
            @param num_row: Number of row to be displayed in the grid
        """
        from torchvision.utils import make_grid
        import matplotlib.pyplot as plt

        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=num_row)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()


    @staticmethod
    def __show_images(imgs: list, img_sizes: list):
        import matplotlib.pyplot as plt

        num_cols = 4
        num_rows = len(imgs) // num_cols
        fig = plt.figure(figsize=(img_sizes[1], img_sizes[2]))
        for i in range(0, len(imgs)):
            fig.add_subplot(num_rows, num_cols, i+1)
            plt.imshow(imgs[i])
        plt.show()
