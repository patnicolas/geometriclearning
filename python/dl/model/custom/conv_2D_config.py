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
                 conv_layers_2D_config: List[ConvLayer2DConfig],
                 max_pooling_kernel: int,
                 out_channels: int,
                 activation: nn.Module,
                 ffnn_out_features: List[int],
                 num_classes: int) -> None:
        """
        Constructor for the configuration of a 2D convolutional network
        @param _id: Identifier for the convolutional network
        @type _id: str
        @param input_size: Width and height of the image
        @type input_size: int
        @param conv_layers_2D_config: List of configuration layers
        @type conv_layers_2D_config: List[ConvLayer2DConfig]
        @param max_pooling_kernel: Size of the kernel and stride for the max pooling block
        @type max_pooling_kernel: int
        @param out_channels: Number of output channel for the last, hidden convolutional layer
        @type out_channels: int
        @param activation: Activation used across all convolutional and feed forward layers
        @type activation: torch module
        @param ffnn_out_features: List of output features for the feed forward layers
        @type ffnn_out_features: List[int]
        @param num_classes: Number of classes or labels
        @type num_classes: int
        """
        conv_blocks = []
        input_dim = (input_size, input_size)

        # Build the various convolutional layers
        for idx in range(len(conv_layers_2D_config)):
            is_batch_normalization = True
            has_bias = False
            conf = conv_layers_2D_config[idx]

            conv_2d_block_builder = Conv2DBlockBuilder(
                in_channels=conf.in_channels,
                out_channels=conv_layers_2D_config[idx + 1].in_channels if idx < len(conv_layers_2D_config)-1
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

        # Compute the number of output channels/nodes for the last convolutional layer
        conv_output_shape = conv_blocks[len(conv_blocks) - 1].compute_out_shapes()
        # Compute the number of input featured for the first feed forward layer
        ffnn_input_shape = out_channels * conv_output_shape[0] * conv_output_shape[1]

        # Generate the feed forward layers
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
        # Finally instantiate the convolutional neural model
        self.conv_model = ConvModel(_id, conv_blocks, ffnn_blocks)

    def __repr__(self) -> AnyStr:
        return repr(self.conv_model)
