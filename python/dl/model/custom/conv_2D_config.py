__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dataclasses import dataclass
from typing import List, AnyStr
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder
import torch.nn as nn
from dl.model.conv_model import ConvModel
from dl.block.conv_block import ConvBlock
from dl.block.ffnn_block import FFNNBlock


@dataclass
class ConvLayer2DConfig:
    in_channels: int
    kernel_size: int
    padding_size: int
    stride_size: int


class Conv2DConfig(object):

    def __init__(self,
                 _id: AnyStr,
                 input_size: int,
                 conv_layer_2D_config: List[ConvLayer2DConfig],
                 max_pooling_kernel: int,
                 out_channels: int,
                 activation: nn.Module,
                 ffnn_out_features: List[int],
                 num_classes: int) -> None:
        conv_blocks = []
        input_dim = (input_size, input_size)

        for idx in range(len(conv_layer_2D_config)):
            is_batch_normalization = True
            has_bias = False
            conf = conv_layer_2D_config[idx]

            conv_2d_block_builder = Conv2DBlockBuilder(
                in_channels=conf.in_channels,
                out_channels=conv_layer_2D_config[idx+1].in_channels if idx < len(conv_layer_2D_config)-1
                            else out_channels,
                input_size=input_dim,
                kernel_size=(conf.kernel_size, conf.kernel_size),
                stride=(conf.stride_size, conf.stride_size),
                padding=(conf.padding_size, conf.padding_size),
                batch_norm=is_batch_normalization,
                max_pooling_kernel=max_pooling_kernel,
                activation=activation,
                bias=has_bias)

            input_dim = conv_2d_block_builder.get_conv_layer_out_shape()
            conv_blocks.append(ConvBlock(str(idx + 1), conv_2d_block_builder))

        conv_output_shape = conv_blocks[len(conv_blocks) - 1].compute_out_shapes()
        ffnn_input_shape = out_channels * conv_output_shape[0] * conv_output_shape[1]

        ffnn_blocks = []
        this_in_features = ffnn_input_shape
        for ffnn_dim_idx in range(len(ffnn_out_features)-1):
            ffnn_block = FFNNBlock.build(block_id=f'hidden{ffnn_dim_idx}',
                                         in_features=this_in_features,
                                         out_features=ffnn_out_features[ffnn_dim_idx],
                                         activation=activation)
            ffnn_blocks.append(ffnn_block)
            this_in_features = ffnn_out_features[ffnn_dim_idx]
        last_ffnn_block = FFNNBlock.build(block_id='output',
                                          in_features=this_in_features,
                                          out_features=num_classes,
                                          activation=nn.Softmax(dim=1))
        ffnn_blocks.append(last_ffnn_block)
        self.conv_model = ConvModel(_id, conv_blocks, ffnn_blocks)

    def __repr__(self) -> AnyStr:
        return repr(self.conv_model)
