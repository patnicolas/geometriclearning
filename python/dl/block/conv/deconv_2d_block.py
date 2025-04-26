__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch.nn as nn
from typing import Self, Optional, AnyStr, Dict, Any

from dl.block.neural_block import NeuralBlock
from dl import ConvException


"""    
    Generic de convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution (kernel, Stride, padding)
         Batch normalization (Optional)
         Activation

    Formula to compute output_dim of a de convolutional block given an in_channels
        output_dim = stride*(in_channels -1) - 2*padding + kernel_size
"""


class DeConv2dBlock(NeuralBlock):
    def __init__(self,
                 block_id: Optional[AnyStr],
                 de_conv_2d_module: nn.ConvTranspose2d,
                 batch_norm_module: Optional[nn.BatchNorm2d] = None,
                 activation_module: Optional[nn.Module] = None,
                 drop_out_module: Optional[nn.Dropout2d] = None) -> None:
        """
        Alternate constructor using a pre-configured block and an optional overwriting activation function. If the
        activation function is not specified, the activation function of the convolutional block is used
        @param block_id: Identifier for this de-convolutional block
        @type block_id: str
        @param de_conv_2d_module: De-convolutional layer module for dimension 2
        @type de_conv_2d_module: nn.ConvTranspose2d
        """
        super(DeConv2dBlock, self).__init__(block_id)

        modules_list = [de_conv_2d_module]

        # Add batch normalization if defined
        if batch_norm_module is not None:
            modules_list.append(batch_norm_module)

        # Add activation if defined
        if activation_module is not None:
            modules_list.append(activation_module)

        # Add drop_out is specified
        if drop_out_module is not None:
            modules_list.append(drop_out_module)
        self.modules_list = modules_list


    @classmethod
    def build(cls, block_id: AnyStr, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor for the de convolutional neural block
        @param block_id: Identifier for this de-convolutional block
        @type block_id: str
        @param block_attributes: Attributes for the Convolutional network
        @type block_attributes: Dict[str, nn.Module]
        """
        de_conv_module = nn.ConvTranspose2d(in_channels=block_attributes['conv_layer'].out_channels,
                                            out_channels=block_attributes['conv_layer'].in_channels,
                                            kernel_size=block_attributes['conv_layer'].kernel_size,
                                            stride=block_attributes['conv_layer'].stride,
                                            padding=block_attributes['conv_layer'].padding,
                                            bias=False)
        return cls(block_id,
                   de_conv_2d_module=de_conv_module,
                   batch_norm_module=block_attributes['batch_norm'],
                   activation_module=block_attributes['activation'],
                   drop_out_module=block_attributes['drop_out'])

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        """
        Cannot build an inverted de-convolutional neural block.
        @param extra: Extra module to be added to the inverted neural structure
        @type extra: nn.Module
        @return: ConvException
        """
        raise ConvException('Cannot invert a de-convolutional neural block')

    def __str__(self) -> AnyStr:
        modules_str = self.__repr__()
        return f'\nConfiguration {self.block_id}:\nModules:\n{modules_str}'

    def __repr__(self) -> str:
        return ' '.join([f'\n{idx}: {str(module)}' for idx, module in enumerate(self.modules_list)])
