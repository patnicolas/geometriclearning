__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from abc import ABC

from dl.block.builder.conv_output_size import SeqConvOutputSize
from dl.block.ffnn_block import FFNNBlock
from dl.block.conv_block import ConvBlock
from dl.model.neural_model import NeuralModel
from dl.model.ffnn_model import FFNNModel
from dl.model.deconv_model import DeConvModel
from typing import List, AnyStr, Dict, Any, Self, Optional, Tuple
import torch
import torch.nn as nn
import logging

from dl.training.neural_net_training import NeuralNetTraining

logger = logging.getLogger('dl.model.ConvModel')

__all__ = ['ConvModel']

"""
    Generic Convolutional neural network which can be used as Gan discriminator or 
    VariationalNeuralBlock Auto-encoder_model decoder_model module. For Gan and LinearVAE, 
    the fully connected linear modules are not defined.
    The build method creates a convolutional neural network without fully connected layers for
    VAE or GAN encoders.
    IF connected layers => CNN
    ELSE: Encoder for VAE or GAN
"""


class ConvModel(NeuralModel, ABC):
    def __init__(self,
                 input_size: int | Tuple[int ,int],
                 model_id: AnyStr,
                 conv_blocks: List[ConvBlock],
                 ffnn_blocks: Optional[List[FFNNBlock]] = None,
                 execution: Optional[NeuralNetTraining] = None) -> None:
        """
        Constructor for this convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param conv_blocks: List of Convolutional Neural Blocks
        @type conv_blocks: List[ConvBlock]
        @param ffnn_blocks: List of Feed-Forward Neural Blocks
        @type ffnn_blocks: List[FFNNBlock]
        """
        ConvModel.is_valid(conv_blocks, ffnn_blocks, input_size)

        self.input_size = input_size
        self.conv_blocks = conv_blocks

        # Record the number of input and output features from the first and last neural block respectively
        self.in_features = conv_blocks[0].conv_block_config.in_channels
        self.out_features = ffnn_blocks[-1].out_features if ffnn_blocks is not None \
            else conv_blocks[-1].conv_block_config.out_channels

        # Define the sequence of modules from the layout
        modules: List[nn.Module] = [module for block in conv_blocks for module in block.modules]

        # If fully connected are provided as CNN
        if ffnn_blocks is not None:
            ffnn_input_size = ConvModel.__linear_layer_input_size(conv_blocks[-1], self.input_size)
            modules.append(nn.Flatten())
            linear_block = FFNNBlock.build(ffnn_blocks[0].block_id,
                                           ffnn_input_size,
                                           ffnn_blocks[0].out_features,
                                           ffnn_blocks[0].activation)
            self.ffnn_blocks = [linear_block]
            if len(ffnn_blocks) > 1:
                [self.ffnn_blocks.append(ffnn_blocks[index]) for index in range(1, len(ffnn_blocks))]
            [modules.append(module) for block in self.ffnn_blocks for module in block.modules]
        else:
            self.ffnn_blocks = None
        super(ConvModel, self).__init__(model_id, nn.Sequential(*modules), execution)

    @classmethod
    def build(cls, model_id: AnyStr, conv_blocks: List[ConvBlock]) -> Self:
        """
        Create a pure convolutional neural network as a convolutional encoder for
        variational auto-encoder or generative adversarial network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param conv_blocks: List of convolutional blocks
        @type conv_blocks: List[ConvBlock]
        @return: Instance of decoder of type ConvModel
        @rtype: ConvModel
        """
        return  cls(model_id, conv_blocks = conv_blocks, ffnn_blocks = None)

    def transpose(self, extra: nn.Module = None) -> DeConvModel:
        """
         Build a de-convolutional neural model from an existing convolutional nodel
         @param extra: Extra module to be added to the inverted neural structure
         @type extra: nn.Module
         @return: Instance of de convolutional model
         @rtype: DeConvModel
         """
        de_conv_blocks = [conv_block.transpose() if idx > 0
                          else conv_block.transpose(extra)
                          for idx, conv_block in enumerate(self.conv_blocks)]
        de_conv_blocks.reverse()
        return DeConvModel(model_id=f'de_{self.model_id}', de_conv_blocks=de_conv_blocks)

    def get_in_features(self) -> int:
        """
        Polymorphic method to retrieve the number of input features
        @return: Number of input features
        @rtype: int
        """
        return self.conv_blocks[0].conv_block_builder.in_channels

    def get_conv_output_size(self) -> int | Tuple[int, int]:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of output features
        @rtype: int
        """
        last_conv_out_channels = self.conv_blocks[-1].conv_block_builder.out_channels
        conv_output_sizes = [conv_block.get_conv_output_size() for conv_block in self.conv_blocks]
        seq_conv_output_size = SeqConvOutputSize(conv_output_sizes)

        return seq_conv_output_size(self.get_in_features(), last_conv_out_channels)

    def has_fully_connected(self) -> bool:
        """
        Test if this convolutional neural network as a fully connected network
        @return: True if at least one fully connected layer exists, False otherwise
        @rtype: bool
        """
        return len(self.ffnn_blocks) > 0

    @staticmethod
    def reshape(x: torch.Tensor, resize: int) -> torch.Tensor:
        """
        Reshape the output of the latest convolutional layer prior to the fully connect layer
        @param x: Tensor for the last conv layer
        @type x: Torch tensor
        @param resize: Number of units of the last conv. layer
        @type resize: TInt
        @return:
        @rtype: Torch tensor
        """
        return x.view(resize, -1)

    def list_modules(self, index: int = 0) -> AnyStr:
        modules = [f'{idx+index}: {str(module)}' for idx, module in enumerate(self.get_modules())]
        return '\n'.join(modules)

    def __repr__(self) -> str:
        return f'State:{self._state_params()}\nModules:\n{self.list_modules(0)}'

    def _state_params(self) -> Dict[AnyStr, Any]:
        dff_model_input_size = self.ffnn_blocks[0].in_features if self.ffnn_blocks is not None else -1
        return {
            "model_id": self.model_id,
            "input_size": self.input_size,
            "output_size": self.out_features,
            "dff_model_input_size": dff_model_input_size
        }

    @staticmethod
    def is_valid(conv_blocks: List[ConvBlock], ffnn_blocks: List[FFNNBlock], input_size: int | Tuple[int, int]) -> bool:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type conv_blocks: List[ConvBlock]
        @param ffnn_blocks:  List of neural blocks which layout is to be evaluated
        @type ffnn_blocks: List[FFNNBlock]
        @param input_size: Input size as int (1D) or Tuple (2D)
        """
        try:
            assert conv_blocks, 'This convolutional model has not defined neural blocks'
            ConvModel.__validate(conv_blocks, input_size)
            if not ffnn_blocks:
                FFNNModel.is_valid(ffnn_blocks)
            return True
        except AssertionError as e:
            logging.error(e)
            return False

    """ ----------------------------   Private helper methods --------------------------- """
    @staticmethod
    def __linear_layer_input_size(last_conv_block: ConvBlock, input_size: int) -> int:
        conv_output_size = last_conv_block.get_conv_output_size()
        conv_output_sizes = conv_output_size(input_size=input_size)
        return last_conv_block.conv_block_config.out_channels * conv_output_sizes[0] * conv_output_sizes[1]


    @staticmethod
    def __validate(neural_blocks: List[ConvBlock], input_size: int | Tuple[int, int]):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"

        for index in range(len(neural_blocks) - 1):
            # Validate the in-channel and out-channels
            next_in_channels = neural_blocks[index + 1].conv_block_config.in_channels
            this_out_channels = neural_blocks[index].conv_block_config.out_channels
            assert next_in_channels == this_out_channels, \
                f'Layer {index} input_tensor != layer {index+1} output'

            this_output_shape = neural_blocks[index].get_conv_output_size()
            next_input_shape = input_size
            assert this_output_shape == next_input_shape, \
                f'This output shape {str(this_output_shape)} should = next input shape {str(next_input_shape)}'


