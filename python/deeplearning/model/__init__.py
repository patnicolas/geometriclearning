__author__ = "Patrick R. Nicolas"
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

import torch
__all__ = ['GrayscaleToRGB', 'OneHotEncoder']

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
