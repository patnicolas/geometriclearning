__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch.nn as nn
from typing import List, Tuple, AnyStr
from dl.block.convblock import ConvBlock
from dl.dlexception import DLException
from dataclasses import dataclass


@dataclass
class ConvParameters:
    kernel_size: int | Tuple[int]
    stride: int | Tuple[int]
    padding: int | Tuple[int]
    batch_norm: int
    activation: nn.Module

    def __str__(self) -> AnyStr:
        f'Kernel size: {self.kernel_size}\nStride: {self.stride}\nPadding: {self.padding}' \
        f'\nBatch normalization: {self.batch_norm}\nActivation function: {str(self.activation)}'



"""
    Object (static) to evaluate the property of a convolutional PyTorch module. All methods are static
"""


class ConvModulesHelper(object):
    @staticmethod
    def is_conv(module: nn.Module) -> bool:
        """
            Test if this module is a convolutional layer
            :param module: Torch module
            :return: True if this is a convolutional layer, False otherwise
        """
        return isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d)

    @staticmethod
    def is_de_conv(module: nn.Module) -> bool:
        """
        Test if this module is a convolutional layer
        @aram module: Proposed torch module
            :return: True if this is a convolutional layer, False otherwise
        """
        return isinstance(module, nn.ConvTranspose1d) or \
            isinstance(module, nn.ConvTranspose2d) or \
            isinstance(module, nn.ConvTranspose3d)

    @staticmethod
    def is_batch_norm(module: nn.Module) -> bool:
        """
             Test if this module is a batch normalization layer
             :param module: Torch module
             :return: True if this is a batch normalization layer, False otherwise
         """
        return isinstance(module, nn.BatchNorm2d) or \
            isinstance(module, nn.BatchNorm1d) or \
            isinstance(module, nn.BatchNorm3d)

    @staticmethod
    def is_activation(module: nn.Module):
        """
        Test if this module is an activation layer
        @param module: Torch module to be evaluated
        @type module: nn.Module
        @return: True if this is a activation layer, False otherwise
        @rtype: bool
         """
        return isinstance(module, nn.ReLU) or \
            isinstance(module, nn.LeakyReLU) or \
            isinstance(module, nn.Tanh) or \
            isinstance(module, nn.Sigmoid)

    @staticmethod
    def get_conv_params(conv_block: ConvBlock, updated_activation: nn.Module) -> ConvParameters(int, int, int, bool, nn.Module):
        conv_modules = list(conv_block.modules)
        # Extract the various components of the convolutional neural block
        batch_norm, activation = ConvModulesHelper.extract_conv_modules(conv_modules)
        # This override the activation function for the output layer, if necessary
        if updated_activation is not None:
            activation = updated_activation

        match conv_block.conv_dimension:
            case 1:
                kernel_size = conv_modules[0].kernel_size
                stride = conv_modules[0].stride
                padding = conv_modules[0].padding
            case 2:
                kernel_size, _ = conv_modules[0].kernel_size
                stride, _ = conv_modules[0].stride
                padding = conv_modules[0].padding
            case _:
                raise DLException(f'Dimension {conv_block.conv_dimension} not supported for deconvolution')

        return ConvParameters(kernel_size, stride, padding, batch_norm, activation)

    @staticmethod
    def extract_conv_modules(conv_modules: List[nn.Module]) -> (nn.Module, nn.Module):
        """
        Extract convolutional layer, batch normalization and activation function from a neural block
        @param conv_modules: Modules defined in this neural block
        @type conv_modules: List of Torch modules
        @return: Pair batch normalization and activation function modules
        @rtype: Pair nn.Module
        """
        activation_function = None
        batch_norm_module = None
        for conv_module in conv_modules:
            if ConvModulesHelper.is_batch_norm(conv_module):
                batch_norm_module = conv_module
            elif ConvModulesHelper.is_activation(conv_module):
                activation_function = conv_module
        return batch_norm_module, activation_function
