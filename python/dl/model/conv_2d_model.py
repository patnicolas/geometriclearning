__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, List, Optional, Tuple

from dl.model.conv_model import ConvModel
from dl.model.mlp_model import MLPBuilder
from dl.model.neural_model import NeuralBuilder
from dl.block.conv import Conv2DataType
from dl.block.conv.conv_2d_block import Conv2dBlock
from dl.block.mlp_block import MLPBlock
import torch.nn as nn


class Conv2dModel(ConvModel):
    def __init__(self,
                 model_id: AnyStr,
                 input_size: Conv2DataType,
                 conv_blocks: List[Conv2dBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        super(Conv2dModel, self).__init__(model_id, input_size, conv_blocks, mlp_blocks)


class Conv2dBuilder(NeuralBuilder):
    keys = ['input_size', 'in_channels_list', 'kernel_size', 'stride',
            'padding', 'is_batch_norm', 'max_pool_kernel', 'activation',
            'in_features_list', 'output_activation', 'bias', 'drop_out']

    def __init__(self, model_id: AnyStr) -> None:
        super(Conv2dBuilder, self).__init__(model_id, Conv2dBuilder.keys)
        # Provide default values that may be overwritten.
        self._attributes['stride'] = (1, 1)
        self._attributes['padding'] = (1, 1)
        self._attributes['is_batch_norm'] = True
        self._attributes['activation'] = nn.ReLU()
        self._attributes['bias'] = False
        self._attributes['drop_out'] = 0.0

    def build(self) -> Conv2dModel:
        # Instantiate the model from the dictionary of
        # Configuration parameters
        model_id = self._attributes['model_id']
        # Generate the convolutional neural blocks from the configuration attributes dictionary
        conv_blocks = self.__create_conv_blocks()
        # Generate the fully connected blocks from the configuration attributes dictionary
        mlp_blocks = self.__create_mlp_blocks()
        # Validation
        Conv2dBuilder.__validate(conv_blocks, mlp_blocks, self._attributes['input_size'])
        return Conv2dModel(model_id, self._attributes['input_size'], conv_blocks, mlp_blocks)

    def __create_conv_blocks(self) -> List[Conv2dBlock]:
        conv_2d_blocks = []
        in_channel = self._attributes['in_channels_list'][0]
        for idx in range(1, len(self._attributes['in_channels_list'])):
            conv_block = Conv2dBlock.build(
                block_id=f'{self._attributes["model_id"]}-{idx}',
                in_channels=in_channel,
                out_channels=self._attributes['in_channels_list'][idx],
                kernel_size=self._attributes['kernel_size'],
                stride=self._attributes['stride'],
                padding=self._attributes['padding'],
                batch_norm=self._attributes['is_batch_norm'],
                max_pooling_kernel=self._attributes['max_pool_kernel'],
                activation=self._attributes['activation'],
                bias=self._attributes['bias'],
                drop_out=self._attributes['drop_out'])
            conv_2d_blocks.append(conv_block)
            in_channel = self._attributes['in_channels_list'][idx]
        return conv_2d_blocks

    def __create_mlp_blocks(self) -> List[MLPBlock]:
        mlp_builder = MLPBuilder('MLP Builder')
        mlp_builder.set(key='in_features_list', value=self._attributes['in_features_list'])
        mlp_builder.set(key='activation', value=self._attributes['activation'])
        mlp_builder.set(key='drop_out', value=self._attributes['drop_out'])
        mlp_builder.set(key='output_activation', value=self._attributes['output_activation'])
        mlp_model = mlp_builder.build()
        return mlp_model.neural_blocks


    @staticmethod
    def __validate(conv_blocks: List[Conv2dBlock],
                   mlp_blocks: List[MLPBlock],
                   input_size: Conv2DataType) -> None:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type conv_blocks: List[ConvBlock]
        @param mlp_blocks:  List of neural blocks which layout is to be evaluated
        @type mlp_blocks: List[MLPBlock]
        @param input_size: Input size as int (1D) or Tuple (2D)
        """
        from dl.model.mlp_model import MLPBuilder
        assert conv_blocks, 'This convolutional model has not defined neural blocks'
        Conv2dBuilder.validate_conv(conv_blocks, input_size)
        MLPBuilder.validate(mlp_blocks)

    """ ----------------------------   Private helper methods --------------------------- """

    def __linear_layer_input_size(self, last_conv_block: Conv2dBlock) -> int:
        from dl.block.conv.conv_output_size import SeqConvOutputSize

        conv_block_sizes = [conv_block.get_conv_output_size() for conv_block in self.conv_blocks]
        conv_model_output_sizes = SeqConvOutputSize(conv_block_sizes)
        conv_output_sizes = conv_model_output_sizes(input_size=self.input_size)
        return last_conv_block.out_channels * conv_output_sizes[0] * conv_output_sizes[1]

    @staticmethod
    def validate_conv(neural_blocks: List[Conv2dBlock], input_size: Conv2DataType) -> None:
        assert len(neural_blocks) > 0, \
            "Convolutional network needs at least one layer"

        for index in range(len(neural_blocks) - 1):
            # 1. Validate the in-channel and out-channels
            next_in_channels = neural_blocks[index + 1].get_in_channels()
            this_out_channels = neural_blocks[index].get_out_channels()
            assert next_in_channels == this_out_channels, \
                f'Layer {index} input_tensor != layer {index + 1} output'

            # 2. Validate the in-channel and out-channels
            this_output_shape_1 = Conv2dBuilder.__get_out_shape_dim(
                conv_block=neural_blocks[index],
                input_size=input_size,
                dim=0
            )
            this_output_shape_2 = Conv2dBuilder.__get_out_shape_dim(
                conv_block=neural_blocks[index],
                input_size=input_size,
                dim=1
            )
            next_input_shape = input_size
            assert (this_output_shape_1 == next_input_shape[0] and
                    this_output_shape_2 == next_input_shape[1],
                    (f'This output shape {str(this_output_shape_1)} should equal '
                     f'next input shape {str(next_input_shape[0])}'))

    @staticmethod
    def __get_out_shape_dim(conv_block: Conv2dBlock, input_size: Conv2DataType, dim: int) -> int:
        stride = conv_block.modules[0].stride[dim]
        padding = conv_block.modules[0].padding[dim]
        kernel_size = conv_block.modules[0].kernel_size[dim]
        num = (input_size[dim] + 2 * padding - kernel_size)
        return int(num / stride) + 1

    @staticmethod
    def output_size(input_size: Tuple[int, int],
                    kernel_size: Tuple[int, int],
                    padding: Tuple[int, int],
                    stride: Tuple[int, int]) -> Tuple[int, int]:
        num_w = (input_size[0] + 2 * padding[0] - kernel_size[0])
        num_h = (input_size[1] + 2 * padding[1] - kernel_size[1])
        return int(num_w / stride[0]) + 1, int(num_h / stride[1]) + 1

