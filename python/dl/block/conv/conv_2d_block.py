__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.conv.conv_block import ConvBlock
from dl.block.conv.deconv_2d_block import DeConv2dBlock
from dl.block.conv.conv_output_size import ConvOutputSize
from typing import AnyStr, Tuple, Optional, Self, Dict, List
import torch.nn as nn
from dl.block.conv import Conv2DataType
from dl import ConvException


"""
Block for the 2-dimensional convolutional layer with the following components
- Conv2d layer torch module
- Max pooling module 
"""


class Conv2dBlock(ConvBlock):
    valid_modules = ('Conv2d', 'MaxPool2d', 'BatchNorm2d', 'Dropout2d')

    def __init__(self,
                 block_id: AnyStr,
                 conv_layer_module: nn.Conv2d,
                 batch_norm_module: Optional[nn.BatchNorm2d] = None,
                 activation_module: Optional[nn.Module] = None,
                 max_pooling_module: Optional[nn.MaxPool2d] = None,
                 drop_out_module: Optional[nn.Dropout2d] = None) -> None:
        """
        Constructor for a 2-dimension convolutional block
        @param block_id: Identifier for the 2D convolutional block
        @type block_id: str
        @param conv_layer_module: Convolutional layer module for dim 2
        @type conv_layer_module:  Conv2d
        @param max_pooling_module: Maximum pooling layer
        @type max_pooling_module: MaxPool2d
        @param batch_norm_module: Optional Batch Normalization module
        @type batch_norm_module: BatchNorm2d
        @param activation_module: Optional activation function/module
        @type activation_module: nn.Module
        @param drop_out_module: Optional drop out module for training
        @type drop_out_module: Dropout2d
        """

        # The 2-dimensional convolutional layer has to be defined
        modules = [conv_layer_module]

        # Add a batch normalization is provided
        if batch_norm_module is not None:
            modules.append(batch_norm_module)
        # Add an activation function is required
        if activation_module is not None:
            modules.append(activation_module)
        # Add a mandatory max pooling module
        if max_pooling_module is not None:
            modules.append(max_pooling_module)
        # Add a Drop out regularization for training if provided
        if drop_out_module is not None:
            modules.append(drop_out_module)
        super(Conv2dBlock, self).__init__(block_id, modules)

    @classmethod
    def build(cls,
              block_id: AnyStr,
              in_channels: int,
              out_channels: int,
              kernel_size: Conv2DataType,
              stride: Conv2DataType = (1, 1),
              padding: Conv2DataType = (0, 0),
              batch_norm: bool = False,
              max_pooling_kernel: int = -1,
              activation: nn.Module = None,
              bias: bool = False,
              drop_out: float = 0.0) -> Self:
        """
        Alternative constructor for a 2-dimension convolutional block
        @param block_id: Identifier for the block id
        @type block_id: str
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Tuple[int, int]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Tuple[int, int]
        @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
        @type stride: Tuple[int, int]
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        @param drop_out: Regularization term applied if > 0
        @type drop_out: float
        """
        conv_2d_module = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=bias)
        batch_norm_module = nn.BatchNorm2d(out_channels) if batch_norm else None
        max_pooling_module = nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=1, padding=0) \
            if max_pooling_kernel > 0 else None
        drop_out_module = nn.Dropout2d(drop_out) if drop_out > 0.0 \
            else None

        return cls(block_id,
                   conv_2d_module,
                   batch_norm_module,
                   activation,
                   max_pooling_module,
                   drop_out_module)

    def transpose(self, output_activation: Optional[nn.Module] = None) -> DeConv2dBlock:
        """
        Build a de-convolutional neural block from an existing convolutional block
        @param output_activation: Optional last activation function
        @type output_activation: nn.Module
        @return: Instance of 2D de-convolutional block
        @rtype: DeConv2dBlock
        """
        if self.attributes is not None:
            if output_activation is not None:
                self.attributes['activation'] = output_activation
            return DeConv2dBlock.build(block_id=f'de_{self.block_id}', attributes=self.attributes)
        else:
            raise ConvException('Generation of De convolutional block is disabled')

    def get_attributes(self) -> AnyStr:
        return str(self.attributes)

    def get_conv_output_size(self) -> ConvOutputSize:
        conv_layer_module = self.modules[0]
        max_pool_module = self.modules[3]
        return ConvOutputSize(
            conv_layer_module.kernel_size,
            conv_layer_module.stride,
            conv_layer_module.padding,
            max_pool_module.kernel_size)

    def validate(self, attributes: Dict[AnyStr, nn.Module] = None) -> Dict[AnyStr, nn.Module]:
        from dl.block.neural_block import NeuralBlock

        self.attributes = attributes
        for module in self.modules:
            module_type = module.__class__.__name__
            if (module_type not in NeuralBlock.supported_activations
                    and module_type not in Conv2dBlock.valid_modules):
                raise ConvException(f'Type of attribute {module_type} '
                                    f'is not compatible with Conv2dBlock')

            if self.attributes is not None:
                match module_type:
                    case 'Conv2d': self.attributes['conv_layer'] = module
                    case 'BatchNorm2d': self.attributes['batch_norm'] = module
                    case 'MaxPool2d':  self.attributes['max_pool'] = module
                    case 'Dropout2d': self.attributes['drop_out'] = module
                    case _: self.attributes['activation'] = module
        return self.attributes


