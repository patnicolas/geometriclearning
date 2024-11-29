__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch

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

