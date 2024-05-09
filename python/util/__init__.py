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