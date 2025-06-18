__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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