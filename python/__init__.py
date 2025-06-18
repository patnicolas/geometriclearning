
import torch
torch.set_default_dtype(torch.float32)

import os
os.environ['SKIP_SLOW_TESTS'] = '1'
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


def tensor_all_close(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-6) -> bool:
    is_match = True
    if t1.shape == t2.shape:
        diff = torch.abs(t1 - t2)
        for val in diff.view(-1):
            if val > rtol:
                is_match = False
    else:
        is_match = False
    return is_match

