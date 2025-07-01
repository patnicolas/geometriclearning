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
torch.set_default_dtype(torch.float32)
from typing import AnyStr, List

"""
Environment variable to enable/disable unit test for execution
"""
import os
os.environ['SKIP_TESTS_IN_PROGRESS'] = '1'
os.environ['SKIP_SLOW_TESTS'] = '1'
SKIP_REASON = 'Skipping some tests due to environment variable'
import logging
import sys

"""
Set up the proper logging handlers and format
"""
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers
logger.addHandler(handler)


"""
Evaluate if two tensors is almost identical, element-wize
"""
def are_tensors_close(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-6) -> bool:
    """
    Test of two tensors are identical within an error tolerance

    @param t1: First tensor
    @type t1: torch.Tensor
    @param t2: Second tensor
    @type t2: torch.Tensor
    @param rtol: Error tolerance
    @type rtol: float
    @return: True if the two tensors are almost identical, False otherwise
    @rtype: boolean
    """
    assert 1e-15 < rtol < 0.01, f'Error tolerance {rtol} is out of range'

    is_match = t1.shape == t2.shape
    if is_match:
        diff = torch.abs(t1 - t2)
        flatten = diff.reshape(-1)
        for val in flatten:
            if val > rtol:
                is_match = False
    return is_match


"""
    Generic format to print a PyTorch tensor with a given decimals
"""

def pretty_torch(x: torch.Tensor, w: int = 8, d: int = 4) -> AnyStr:
    match x.dim():
        case 0:
            print(f'{x.item():>{w}.{d}f}')
        case 1:
            print(" ".join(f"{_x.item():>{w}.{d}f}" for _x in x))
        case _:
            for _x in x:
                pretty_torch(_x, w, d)



