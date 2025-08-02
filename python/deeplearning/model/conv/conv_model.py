__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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

from abc import ABC

from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.block.conv.conv_block import ConvBlock
from deeplearning.model.neural_model import NeuralModel
from deeplearning.model.mlp.mlp_model import MLPModel
from deeplearning.model.conv.deconv_2d_model import DeConv2dModel
from deeplearning.block.conv import ConvDataType
from typing import List, AnyStr, Dict, Any, Optional
import torch
import torch.nn as nn
import logging
logger = logging.getLogger('deeplearning.model.ConvModel')
__all__ = ['ConvModel']


class ConvModel(NeuralModel, ABC):
    """
        Generic Convolutional neural network which can be used as Gan discriminator or
    VariationalNeuralBlock Auto-encoder_model decoder_model module. For Gan and LinearVAE,
    the fully connected linear modules are not defined.
    The build method creates a convolutional neural network without fully connected layers for
    VAE or GAN encoders.
    IF connected layers => CNN
    ELSE: Encoder for VAE or GAN

    https://patricknicolas.substack.com/p/modular-deep-learning-models-with
    """
    def __init__(self,
                 model_id: AnyStr,
                 input_size: ConvDataType,
                 conv_blocks: List[ConvBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        """
        Constructor for this convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param conv_blocks: List of Convolutional Neural Blocks
        @type conv_blocks: List[ConvBlock]
        @param mlp_blocks: List of Feed-Forward Neural Blocks
        @type mlp_blocks: List[MLPBlock]
        """
        self.input_size = input_size
        self.conv_blocks = conv_blocks
        self.mlp_blocks = mlp_blocks

        # Define the sequence of modules from the layout
        modules = [module for block in self.conv_blocks
                   for module in block.modules_list]

        # If fully connected, MLP layers are included ....
        if self.mlp_blocks is not None:
            modules.append(nn.Flatten())
            # Compute the size of the 1 dimensional input to the first fully
            # connected (Linear) layer
            flatten_input_size = self.__linear_layer_input_size(conv_blocks[-1])
            # Retrieve the first linear layer of the MLP sequence
            first_linear_layer = self.mlp_blocks[0].modules_list[0]
            first_linear_layer.in_features = flatten_input_size

            # Generate the sequence of modules
            [modules.append(module) for block in self.mlp_blocks
             for module in block.modules_list]
        else:
            self.mlp_blocks = None

        super(ConvModel, self).__init__(model_id, nn.Sequential(*modules))

    def transpose(self, extra: nn.Module = None) -> DeConv2dModel:
        """
         Build a de-convolutional neural model from an existing convolutional nodel
         @param extra: Extra module to be added to the inverted neural structure
         @type extra: nn.Module
         @return: Instance of de convolutional model
         @rtype: DeConv2dModel
         """
        de_conv_blocks = [conv_block.transpose() for conv_block in self.conv_blocks]
        de_conv_blocks.reverse()
        return DeConv2dModel(model_id=f'de_{self.model_id}', deconv_blocks=de_conv_blocks)

    def get_in_features(self) -> int:
        """
        Polymorphic method to retrieve the number of input features
        @return: Number of input features
        @rtype: int
        """
        return self.conv_blocks[0].in_channels

    def get_out_features(self) -> int:
        return self.mlp_blocks[-1].out_features if self.mlp_blocks is not None \
            else self.conv_blocks[-1].out_channels

    def get_flatten_output_size(self) -> int:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of output features
        @rtype: int
        """
        flatten_out_size = self.__linear_layer_input_size(self.conv_blocks[-1])
        return flatten_out_size

    def has_fully_connected(self) -> bool:
        """
        Test if this convolutional neural network as a fully connected network
        @return: True if at least one fully connected layer exists, False otherwise
        @rtype: bool
        """
        return len(self.mlp_blocks) > 0

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

    def __str__(self) -> str:
        return f'\nModel: {self.model_id}\nState:{self._state_params()}\nModules:\n{self.list_modules(0)}'

    def __repr__(self) -> str:
        return f'\n{self.list_modules(0)}'

    def _state_params(self) -> Dict[AnyStr, Any]:
        dff_model_input_size = self.mlp_blocks[0].in_features \
            if self.mlp_blocks is not None else -1
        return {
            "model_id": self.model_id,
            "input_size": self.input_size,
            "output_size": self.out_features,
            "dff_model_input_size": dff_model_input_size
        }

    @staticmethod
    def is_valid(conv_blocks: List[ConvBlock], mlp_blocks: List[MLPBlock], input_size: ConvDataType) -> bool:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type conv_blocks: List[ConvBlock]
        @param mlp_blocks:  List of neural blocks which layout is to be evaluated
        @type mlp_blocks: List[MLPBlock]
        @param input_size: Input size as int (1D) or Tuple (2D)
        """
        try:
            assert conv_blocks, 'This convolutional model has not defined neural blocks'
            ConvModel.__validate(conv_blocks, input_size)
            if not mlp_blocks:
                MLPModel.is_valid(mlp_blocks)
            return True
        except AssertionError as e:
            logging.error(e)
            return False

    """ ----------------------------   Private helper methods --------------------------- """
    def __linear_layer_input_size(self, last_conv_block: ConvBlock) -> int:
        from deeplearning.block.conv.conv_output_size import SeqConvOutputSize

        conv_block_sizes = [conv_block.get_conv_output_size() for conv_block in self.conv_blocks]
        conv_model_output_sizes = SeqConvOutputSize(conv_block_sizes)
        conv_output_sizes = conv_model_output_sizes(input_size=self.input_size)
        return last_conv_block.get_out_channels() * conv_output_sizes[0] * conv_output_sizes[1]

    @staticmethod
    def __validate(neural_blocks: List[ConvBlock], input_size: ConvDataType):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"

        for index in range(len(neural_blocks) - 1):
            # Validate the in-channel and out-channels
            next_in_channels = neural_blocks[index + 1].get_in_channels()
            this_out_channels = neural_blocks[index].get_out_channels()
            assert next_in_channels == this_out_channels, \
                f'Layer {index} input_tensor != layer {index+1} output'

            this_output_shape = neural_blocks[index].get_conv_output_size()
            next_input_shape = input_size
            assert this_output_shape == next_input_shape, \
                f'This output shape {str(this_output_shape)} should = next input shape {str(next_input_shape)}'


