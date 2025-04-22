__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.conv.conv_block import ConvBlock
from typing import AnyStr, Tuple, Optional, Self, Dict, Any
import torch.nn as nn
from dl.block.conv import Conv3DataType
from dl import ConvException


class Conv3dBlock(ConvBlock):
    valid_modules = ('Conv3d', 'MaxPool3d', 'BatchNorm3d', 'Dropout3d')

    def __init__(self,
                 block_id: AnyStr,
                 conv_layer_module: nn.Conv3d,
                 batch_norm_module: Optional[nn.BatchNorm3d] = None,
                 activation_module: Optional[nn.Module] = None,
                 max_pooling_module: Optional[nn.MaxPool3d] = None,
                 drop_out_module: Optional[nn.Dropout3d] = None) -> None:
        """
        Constructor for a 3-dimension convolutional block
        @param block_id: Identifier for the 2D convolutional block
        @type block_id: str
        @param conv_layer_module: Convolutional layer module for dim 2
        @type conv_layer_module:  Conv3d
        @param max_pooling_module: Maximum pooling layer
        @type max_pooling_module: MaxPool3d
        @param batch_norm_module: Optional Batch Normalization module
        @type batch_norm_module: BatchNorm3d
        @param activation_module: Optional activation function/module
        @type activation_module: nn.Module
        @param drop_out_module: Optional drop out module for training
        @type drop_out_module: Dropout3d
        """
        super(Conv3dBlock, self).__init__(block_id)

        # The 3-dimensional convolutional layer has to be registered
        modules_list = nn.ModuleList()
        modules_list.append(conv_layer_module)

        # Add the batch normalization if defined
        if batch_norm_module is not None:
            modules_list.append(batch_norm_module)
        # Activation to be added if defined
        if activation_module is not None:
            modules_list.append(activation_module)
        # Added max pooling module
        if max_pooling_module is not None:
            modules_list.append(max_pooling_module)
        # Add drop out for training is defined
        if drop_out_module is not None:
            modules_list.append(drop_out_module)

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        block_id = block_attributes['block_id']
        in_channels = block_attributes['in_channels']
        out_channels = block_attributes['out_channels']
        kernel_size = block_attributes['kernel_size']
        stride = block_attributes['stride']
        padding = block_attributes['padding']
        bias = block_attributes['bias']
        batch_norm_module = block_attributes['batch_norm']
        activation_module = block_attributes['activation']
        max_pooling_module = block_attributes['max_pooling']
        dropout_ratio = block_attributes['dropout_ratio']
        return cls(block_id=block_id,
                   conv_layer_module=nn.Conv3d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               bias=bias),
                   batch_norm_module=batch_norm_module,
                   activation_module=activation_module,
                   max_pooling_module=max_pooling_module,
                   drop_out_module=nn.Dropout3d(dropout_ratio))

    @classmethod
    def build_from_params(cls,
                          block_id: AnyStr,
                          in_channels: int,
                          out_channels: int,
                          kernel_size: Conv3DataType,
                          stride: Conv3DataType = (1, 1, 1),
                          padding: Conv3DataType = (0, 0, 0),
                          batch_norm: bool = False,
                          max_pooling_kernel: int = -1,
                          activation: nn.Module = None,
                          bias: bool = False,
                          drop_out: float = 0.0) -> Self:
        """
        Alternative constructor for a 3-dimension convolutional block
        @param block_id: Identifier for the block id
        @type block_id: str
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 3D
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
        conv_3d_module = nn.Conv3d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=bias)
        batch_norm_module = nn.BatchNorm3d(out_channels) if batch_norm else None
        max_pooling_module = nn.MaxPool3d(kernel_size=max_pooling_kernel, stride=1, padding=0) \
            if max_pooling_kernel > 0 else None
        drop_out_module = nn.Dropout3d(drop_out) if drop_out > 0.0 \
            else None

        return cls(block_id,
                   conv_3d_module,
                   batch_norm_module,
                   activation,
                   max_pooling_module,
                   drop_out_module)

    def validate(self, attributes: Dict[AnyStr, nn.Module] = None) -> Dict[AnyStr, nn.Module]:
        from dl.block.neural_block import NeuralBlock

        self.attributes = attributes
        for module in list(self.modules_list):
            module_type = module.__class__.__name__
            if (module_type not in NeuralBlock.supported_activations
                    and module_type not in Conv3dBlock.valid_modules):
                raise ConvException(f'Type of attribute {module_type} '
                                    f'is not compatible with Conv3dBlock')

            if self.attributes is not None:
                match module_type:
                    case 'Conv3d': self.attributes['conv_layer'] = module
                    case 'BatchNorm3d': self.attributes['batch_norm'] = module
                    case 'MaxPool3d':  self.attributes['max_pool'] = module
                    case 'Dropout3d': self.attributes['drop_out'] = module
                    case _: self.attributes['activation'] = module
        return self.attributes
