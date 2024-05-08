__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

"""
    Generic Convolutional neural network which can be used as Gan discriminator or VariationalNeuralBlock Auto-encoder_model
    decoder_model module. For Gan and LinearVAE, the fully connected linear modules are not defined
"""

from dl.block.ffnnblock import FFNNBlock
from dl.block.convblock import ConvBlock
from dl.model.neuralmodel import NeuralModel
from typing import List, AnyStr, Dict, Any
from util import log_size
import torch
import torch.nn as nn

class ConvModel(NeuralModel):
    def __init__(self,
                 model_id: AnyStr,
                 conv_blocks: List[ConvBlock],
                 ffnn_blocks: List[FFNNBlock]):
        """
        Constructor for this convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param conv_blocks: List of Convolutional Neural Blocks
        @type conv_blocks: List[ConvBlock]
        @param ffnn_blocks: List of Feed-Forward Neural Blocks
        @type ffnn_blocks: List[FFNNBlock]
        """
        assert len(conv_blocks) > 0, 'This convolutional model has not defined neural blcoks'
        self.conv_blocks = conv_blocks
        self.ffnn_blocks = ffnn_blocks

        # Record the number of input and output features from the first and last neural block respectively
        self.in_features = conv_blocks[0].in_features
        self.out_features = ffnn_blocks[-1].out_features
        # Define the sequence of modules from the layout
        conv_modules = [module for block in conv_blocks for module in block.modules]
        ffnn_modules = [module for block in ffnn_blocks for module in block.modules]
        modules = conv_modules + ffnn_modules

        super(ConvModel, self).__init__(model_id, nn.Sequential(*modules))

    def has_fully_connected(self) -> bool:
        """
        Test if this convolutional neural network as a fully connected network
        @return: True if at least one fully connected layer exists, False otherwise
        @rtype: bool
        """
        return self.dff_model is not None

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

    def _state_params(self) -> Dict[AnyStr, Any]:
        return {
            "model_id": self.model_id,
            "conv_dimension": self.conv_blocks[0].conv_dimension,
            "input_size": self.conv_blocks[0].in_channels,
            "output_size": self.ffnn_blocks[-1].out_features ,
            "dff_model_input_size": self.ffnn_blocks[0].in_features
        }

    def __repr__(self) -> str:
        modules = [module for module in self.conv_model.modules() if not isinstance(module, nn.Sequential)]
        conv_repr = ' '.join([f'\n{str(module)}' for module in modules if module is not None])
        if self.dff_model is not None:
            modules = [module for module in self.dff_model.modules() if not isinstance(module, nn.Sequential)]
            dff_repr = ' '.join([f'\n{str(module)}' for module in modules if module is not None])
            return f'{self._state_params()}{conv_repr}{dff_repr}'
        else:
            return f'{self._state_params()}{conv_repr}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
           Process the model as sequence of modules, implicitly called by __call__
           :param x: Input input_tensor
           :return: Tensor output from this network
        """
        log_size(x, 'Input Conv model')
        x = self.conv_model(x)
        log_size(x, 'Output Conv model')
        # If a full connected network is appended to the convolutional layers
        if self.dff_model is not None:
            log_size(x, 'Before width Conv')
            sz = x.shape[0]
            x = ConvModel.reshape(x, sz)
            log_size(x, 'After width Conv')
            x = self.dff_model(x)
            log_size(x, 'Output connected Conv')
        return x




