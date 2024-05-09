__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch.nn as nn
from typing import Tuple
from dl.block.convblock import ConvBlock


class DeConvBlock(nn.Module):
    def __init__(self,
                 conv_dimension: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int],
                 stride: int,
                 padding: int | Tuple[int],
                 batch_norm: bool,
                 activation: nn.Module,
                 bias: bool):
        ConvBlock.validate_input(conv_dimension, in_channels, out_channels, kernel_size, stride, 1)
