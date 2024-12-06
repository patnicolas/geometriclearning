__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch

"""
Convert an image from a 1-channel (Black & White) to 3 channel (RGB), using the mode if image

"""


class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img


class OneHotEncoder(object):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, target):
        return torch.nn.functional.one_hot(target, self.num_classes)


def training_exec_config():
    return None