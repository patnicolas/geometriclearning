__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import logging
import torch
from typing import AnyStr, NoReturn, overload


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


@overload
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

    print("Available modules:")
    for mod in available:
        print(f"  - {mod}")

    if missing:
        print("\nMissing modules:")
        for mod in missing:
            print(f"  - {mod}")
    else:
        print("\nAll modules are available!")