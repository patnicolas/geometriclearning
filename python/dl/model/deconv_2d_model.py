__author__ = "Patrick Nicolas"
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

from dl.model.neural_model import NeuralModel
from dl.model.mlp_model import MLPModel
from dl.block.conv.deconv_2d_block import DeConv2dBlock
from dl.block.mlp_block import MLPBlock
from typing import AnyStr, List, Self, Dict, Any

import torch.nn as nn
import torch
import logging
import python
__all__ = ['DeConv2dModel']


class DeConv2dModel(NeuralModel, ABC):
    def __init__(self,
                 model_id: AnyStr,
                 deconv_blocks: List[DeConv2dBlock]) -> None:
        """
        Constructor for this de-convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param deconv_blocks: List of Convolutional Neural Blocks
        @type deconv_blocks: List[ConvBlock]
        @param ffnn_blocks: Optional list of Feed-Forward Neural Blocks
        @type ffnn_blocks: List[MLPBlock]
        """
        self.deconv_blocks = deconv_blocks

        # Define the sequence of modules from the layout
        de_conv_modules = [module for block in deconv_blocks for module in block.modules]
        super(DeConv2dModel, self).__init__(model_id, nn.Sequential(*de_conv_modules))

    @classmethod
    def build(cls, model_id: AnyStr, de_conv_blocks: List[DeConv2dBlock]) -> Self:
        """
        Create a pure de-convolutional neural network as a convolutional decoder for
        variational auto-encoder or generative adversarial network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param de_conv_blocks: List of de-convolutional blocks
        @type de_conv_blocks: List[DeConvBlock]
        @return: Instance of decoder of type DeConvModel
        @rtype: DeConv2dModel
        """
        return cls(model_id, de_conv_blocks, [])

    def has_fully_connected(self) -> bool:
        """
        Test if this convolutional neural network as a fully connected network
        @return: True if at least one fully connected layer exists, False otherwise
        @rtype: bool
        """
        return len(self.ffnn_blocks) > 1

    def __str__(self) -> str:
        return f'\nModel: {self.model_id}\nState:{self._state_params()}\nModules:\n{self.list_modules(0)}'

    def __repr__(self) -> str:
        return f'\n{self.list_modules(0)}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the model as sequence of modules, implicitly called by __call__
        @param x: Input input_tensor
        @type x: A torch tensor
        @return: A tensor output from last layer
        @rtype; Torch tensor
        """
        logging.debug(x, 'Input Conv model')
        x = self.modules_seq(x)
        logging.debug(x, 'Output Conv model')
        # If a full connected network is appended to the convolutional layers
        if self.ffnn_blocks is not None and len(self.ffnn_blocks) > 0:
            logging.debug(x, 'Before width Conv')
            sz = x.shape[0]
            x = DeConv2dModel.reshape(x, sz)
            logging.debug(x, 'After width Conv')
            x = self.dff_model(x)
            logging.debug(x, 'Output connected Conv')
        return x

    def list_modules(self, index: int = 0) -> AnyStr:
        modules = [f'{idx + index}: {str(module)}' for idx, module in enumerate(self.get_modules())]
        return '\n'.join(modules)

    def _state_params(self) -> Dict[AnyStr, Any]:
        return {
            "model_id": self.model_id,
            "input_size": self.deconv_blocks[0].conv_block_config.in_channels,
            "output_size": self.ffnn_blocks[-1].out_features if self.ffnn_blocks is not None else -1,
            "dff_model_input_size": self.ffnn_blocks[0].in_features if self.ffnn_blocks is not None else -1
        }

    @staticmethod
    def is_valid(de_conv_blocks: List[DeConv2dBlock], ffnn_blocks: List[MLPBlock]) -> bool:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param de_conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type de_conv_blocks: List[ConvBlock]
        @param ffnn_blocks:  List of neural blocks which layout is to be evaluated
        @type ffnn_blocks: List[MLPBlock]
        """
        try:
            assert de_conv_blocks, 'This convolutional model has not defined neural blocks'
            DeConv2dModel.__validate(de_conv_blocks)
            if not ffnn_blocks:
                MLPModel.is_valid(ffnn_blocks)
            return True
        except AssertionError as e:
            logging.error(e)
            return False

    """ ----------------------------   Private helper methods --------------------------- """

    @staticmethod
    def __validate(neural_blocks: List[DeConv2dBlock]):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"
        for index in range(len(neural_blocks) - 1):
            assert neural_blocks[index + 1].in_channels == neural_blocks[index].out_channels, \
                f'Layer {index} input_tensor != layer {index + 1} output'
