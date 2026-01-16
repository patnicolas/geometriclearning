__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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

# Standard Library imports
from typing import AnyStr, Tuple, Optional, Self, Dict, Any, List
# 3rd Party imports
import torch.nn as nn
# Library imports
from deeplearning.block.conv.conv_block import ConvBlock
from deeplearning.block.conv.deconv_2d_block import DeConv2dBlock
from deeplearning.block.conv.conv_output_size import ConvOutputSize
from deeplearning.block.conv import Conv2DataType
from deeplearning import ConvException
__all__ = ['Conv2dBlock']


class Conv2dBlock(ConvBlock):
    """
    A neural block is a foundational unit which inherits from PyTorch class, Module and encapsulates a set of
    PyTorch modules as described below in the case of the 2-dimensional convolutional layer:
    - Conv2d layer torch module
    - Max pooling module
    - Activation module
    - Optional batch normalization
    - Optional dropout regularization for training

    A Neural block can be constructor directly from PyTorch modules (nn.Module) using the default constructor
    or from a descriptive dictionary of block attributes such as
     {
            'block_id': 'my_block',
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': (3, 3),
            'stride': (1, 1),
            'padding': (2, 2),
            'bias': True,
            'batch_norm': nn.BatchNorm2d(32),
            'activation': nn.ReLU(),
            'max_pooling': nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            'dropout_ratio': 0.3
        }

    Reference: https://patricknicolas.substack.com/p/reusable-neural-blocks-in-pytorch
    """
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
        super(Conv2dBlock, self).__init__(block_id)

        # The 2-dimensional convolutional layer has to be defined
        modules_list = nn.ModuleList()
        modules_list.append(conv_layer_module)

        # Add a batch normalization is provided
        if batch_norm_module is not None:
            modules_list.append(batch_norm_module)
        # Add an activation function is required
        if activation_module is not None:
            modules_list.append(activation_module)
        # Add a mandatory max pooling module
        if max_pooling_module is not None:
            modules_list.append(max_pooling_module)
        # Add a Drop out regularization for training if provided
        if drop_out_module is not None:
            modules_list.append(drop_out_module)
        self.modules_list = modules_list

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor that instantiates a 2-dimensional neural block from a dictionary
        of neural attributes
        @param block_attributes: Dictionary of neural attributes
        @type block_attributes: Dict[AnyStr, Any]
        @return: Instance of a Conv2dBlock
        @rtype: Conv2dBlock
        """
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
                   conv_layer_module=nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               bias=bias),
                   batch_norm_module=batch_norm_module,
                   activation_module=activation_module,
                   max_pooling_module=max_pooling_module,
                   drop_out_module=nn.Dropout2d(dropout_ratio))

    @classmethod
    def build_from_params(cls,
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

        return cls(block_id=block_id,
                   conv_layer_module=conv_2d_module,
                   batch_norm_module=batch_norm_module,
                   activation_module=activation,
                   max_pooling_module=max_pooling_module,
                   drop_out_module=drop_out_module)

    def reset_parameters(self) -> None:
        self.modules_list[0].reset_parameters()

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
            self.attributes['block_id'] = f'de_{self.block_id}'
            return DeConv2dBlock.build(block_attributes=self.attributes)
        else:
            raise ConvException('Generation of De convolutional block is disabled')

    def get_attributes(self) -> AnyStr:
        return str(self.attributes)

    def get_conv_output_size(self) -> ConvOutputSize:
        conv_layer_module = self.modules_list[0]
        max_pool_module = self.modules_list[3]
        return ConvOutputSize(
            conv_layer_module.kernel_size,
            conv_layer_module.stride,
            conv_layer_module.padding,
            max_pool_module.kernel_size)

    def get_flatten_output_size(self, input_size: int | Tuple[int, int], conv_blocks: List[ConvBlock]):
        from deeplearning.block.conv.conv_output_size import SeqConvOutputSize

        conv_block_sizes = [conv_block.get_conv_output_size() for conv_block in conv_blocks]
        conv_model_output_sizes = SeqConvOutputSize(conv_block_sizes)
        conv_output_sizes = conv_model_output_sizes(input_size=input_size)
        return self.get_out_channels() * conv_output_sizes[0] * conv_output_sizes[1]

    def validate(self, attributes: Dict[AnyStr, nn.Module] = None) -> Dict[AnyStr, nn.Module]:
        from deeplearning.block.neural_block import NeuralBlock

        self.attributes = attributes
        for module in list(self.modules_list):
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


