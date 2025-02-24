__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, List, Optional, Self
from dl.model.conv_model import ConvModel
from dl.block.conv import Conv2DataType
from dl.block.conv.conv_2d_block import Conv2dBlock
from dl.block.ffnn_block import FFNNBlock
import torch.nn as nn


class Conv2dModel(ConvModel):
    def __init__(self,
                 model_id: AnyStr,
                 input_size: Conv2DataType,
                 conv_blocks: List[Conv2dBlock],
                 ffnn_blocks: Optional[List[FFNNBlock]] = None) -> None:
        super(Conv2dModel, self).__init__(model_id, input_size, conv_blocks, ffnn_blocks)

    @classmethod
    def build(cls,
              model_id: AnyStr,
              input_size: Conv2DataType,
              in_channels: List[int],
              kernel_size: Conv2DataType,
              stride: Conv2DataType,
              padding: Conv2DataType,
              batch_norm: bool,
              max_pooling_kernel: int,
              activation: nn.Module = nn.ReLU(),
              in_features: List[int] = None,
              output_activation: nn.Module = None,
              bias: bool = False,
              drop_out: float = 0.2) -> Self:

        from dl.model.ffnn_model import FFNNModel

        conv_2d_blocks = []
        in_channel = in_channels[0]
        for idx in range(1, len(in_channels)):
            conv_block = Conv2dBlock.build(
                            block_id=f'{model_id}-{idx}',
                            in_channels=in_channel,
                            out_channels=in_channels[idx],
                            kernel_size=kernel_size,
                            deconvolution_enabled=False,
                            stride=stride,
                            padding=padding,
                            batch_norm=batch_norm,
                            max_pooling_kernel=max_pooling_kernel,
                            activation=activation,
                            bias=bias,
                            drop_out=drop_out)
            conv_2d_blocks.append(conv_block)
            in_channel = in_channels[idx]

        ffnn_blocks = FFNNModel.create_ffnn_blocks(
                    model_id,
                    in_features,
                    activation,
                    drop_out,
                    output_activation) \
            if in_features is not None \
            else None
        return cls(model_id, input_size, conv_2d_blocks, ffnn_blocks)
