__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from abc import ABC

from dl.block.ffnnblock import FFNNBlock
from dl.block.convblock import ConvBlock
from dl.model.neuralmodel import NeuralModel
from dl.model.ffnnmodel import FFNNModel
from typing import List, AnyStr, Dict, Any, Self, Optional
from util import log_size
import torch
import torch.nn as nn
import logging
logger = logging.getLogger('dl.model.ConvModel')

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
                 model_id: AnyStr,
                 conv_blocks: List[ConvBlock],
                 ffnn_blocks: Optional[List[FFNNBlock]]):
        """
        Constructor for this convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param conv_blocks: List of Convolutional Neural Blocks
        @type conv_blocks: List[ConvBlock]
        @param ffnn_blocks: List of Feed-Forward Neural Blocks
        @type ffnn_blocks: List[FFNNBlock]
        """
        ConvModel.is_valid(conv_blocks, ffnn_blocks)
        self.conv_blocks = conv_blocks

        # Record the number of input and output features from the first and last neural block respectively
        self.in_features = conv_blocks[0].in_channels
        self.out_features = ffnn_blocks[-1].out_features
        # Define the sequence of modules from the layout
        conv_modules = [module for block in conv_blocks for module in block.modules]

        # If fully connected are provided as CNN
        if ffnn_blocks:
            self.ffnn_blocks = ffnn_blocks
            ffnn_modules = [module for block in ffnn_blocks for module in block.modules]
            modules = conv_modules + [nn.Flatten()] + ffnn_modules
        else:
            modules = conv_modules
        super(ConvModel, self).__init__(model_id, nn.Sequential(*modules))

    @classmethod
    def build(cls, model_id: AnyStr, conv_blocks: List[ConvBlock]) -> Self:
        """
        Create a pure convolutional neural network as a convolutional encoder for
        variational auto-encoder or generative adversarial network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param conv_blocks: List of convolutional blocks
        @type de_conv_blocks: List[ConvBlock]
        @return: Instance of decoder of type ConvModel
        @rtype: ConvModel
        """
        return  cls(model_id, conv_blocks, [])

    def has_fully_connected(self) -> bool:
        """
        Test if this convolutional neural network as a fully connected network
        @return: True if at least one fully connected layer exists, False otherwise
        @rtype: bool
        """
        return self.ffnn_blocks

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

    def __repr__(self) -> str:
        modules = [str(module) for module in self.get_modules()]
        module_repr = '\n'.join(modules)
        return f'State:{self._state_params()}\nModules:\n{module_repr}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the model as sequence of modules, implicitly called by __call__
        @param x: Input input_tensor
        @type x: A torch tensor
        @return: A tensor output from last layer
        @rtype; Torch tensor
        """
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
        """
        log_size(x, 'Input Conv model')
        x = self.model(x)
        log_size(x, 'Output Conv model')
        return x

    def _state_params(self) -> Dict[AnyStr, Any]:
        return {
            "model_id": self.model_id,
            "conv_dimension": self.conv_blocks[0].conv_dimension,
            "input_size": self.conv_blocks[0].in_channels,
            "output_size": self.ffnn_blocks[-1].out_features ,
            "dff_model_input_size": self.ffnn_blocks[0].in_features
        }

    @staticmethod
    def is_valid(conv_blocks: List[ConvBlock], ffnn_blocks: List[FFNNBlock]) -> bool:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type conv_blocks: List[ConvBlock]
        @param ffnn_blocks:  List of neural blocks which layout is to be evaluated
        @type ffnn_blocks: List[FFNNBlock]
        """
        try:
            assert conv_blocks, 'This convolutional model has not defined neural blocks'
            ConvModel.__validate(conv_blocks)
            if not ffnn_blocks:
                FFNNModel.is_valid(ffnn_blocks)
            return True
        except AssertionError as e:
            logging.error(e)
            return False

    """ ----------------------------   Private helper methods --------------------------- """
    @staticmethod
    def __validate(neural_blocks: List[ConvBlock]):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"
        for index in range(len(neural_blocks) - 1):
            assert neural_blocks[index + 1].in_channels == neural_blocks[index].out_channels, \
                f'Layer {index} input_tensor != layer {index+1} output'