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

import torch
from typing import AnyStr, NoReturn, overload
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers
logger.addHandler(handler)


@overload
def log_size(x: torch.Tensor, comment: AnyStr = "") -> NoReturn:
    """
    Utility to display the shape of an input_tensor
    @param x: Torch input_tensor which size to be computed
    @type x: Torch Tensor
    @param comment: Optional comments
    @type comment: AnyStr
    """
    assert isinstance(x, torch.Tensor), '\nNot a Tensor type'
    sz = list(x.size())
    logging.info(f'{str(sz)} {comment}')


def log_size(x: torch.Tensor, y: torch.Tensor, comment: AnyStr = '') -> NoReturn:
    """
    Utility to display the shape of two input_tensor
    @param x: Torch input_tensor which size to be computed
    @type x: Torch Tensor
    @param y: Second input_tensor which size to be computed
    @type y: Torch Tensor
    @param comment: Optional comments
    @type comment: AnyStr
    """
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), '\nNot a Tensor type'
    szx = list(x.size())
    szy = list(y.size())
    logging.info(f'{str(szx)} {str(szy)} {comment}')


from typing import List
import importlib


def check_modules_availability(modules: List[AnyStr]) -> None:
    available = []
    missing = []

    for name in modules:
        if importlib.util.find_spec(name):
            available.append(name)
        else:
            missing.append(name)

    logging.info("Available modules:")
    for mod in available:
        logging.info(f"  - {mod}")

    if missing:
        logging.info("\nMissing modules:")
        for mod in missing:
            logging.info(f"  - {mod}")
    else:
        logging.info("\nAll modules are available!")