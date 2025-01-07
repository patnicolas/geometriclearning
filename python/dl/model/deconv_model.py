__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from abc import ABC

from dl.model.neural_model import NeuralModel
from dl.model.ffnn_model import FFNNModel
from dl.block.cnn.deconv_2d_block import DeConv2DBlock
from dl.block.ffnn_block import FFNNBlock
from typing import AnyStr, List, Optional, Self, Dict, Any

from dl.training.neural_training import NeuralTraining
import torch.nn as nn
import torch
import logging
logger = logging.getLogger('dl.model.DeConvModel')


class DeConvModel(NeuralModel, ABC):
    def __init__(self,
                 model_id: AnyStr,
                 de_conv_blocks: List[DeConv2DBlock],
                 last_activation: Optional[nn.Module] = None,
                 ffnn_blocks: Optional[List[FFNNBlock]] = None,
                 execution: Optional[NeuralTraining] = None) -> None:
        """
        Constructor for this de-convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param de_conv_blocks: List of Convolutional Neural Blocks
        @type de_conv_blocks: List[ConvBlock]
        @param ffnn_blocks: Optional list of Feed-Forward Neural Blocks
        @type ffnn_blocks: List[FFNNBlock]
        """
        self.de_conv_blocks = de_conv_blocks

        # Record the number of input and output features from the first and last neural block respectively
        self.in_features = de_conv_blocks[0].conv_block_config.in_channels
        self.out_features = ffnn_blocks[-1].out_features if ffnn_blocks is not None \
            else de_conv_blocks[-1].conv_block_config.out_channels

        # Define the sequence of modules from the layout
        de_conv_modules = [module for block in de_conv_blocks for module in block.modules]
        if last_activation is not None:
            de_conv_modules[-1] = last_activation

        # If fully connected are provided as CNN
        if ffnn_blocks is not None:
            ffnn_modules = [module for block in ffnn_blocks for module in block.modules]
            modules = de_conv_modules + [nn.Unflatten] + ffnn_modules
        else:
            modules = de_conv_modules
        self.ffnn_blocks = ffnn_blocks
        super(DeConvModel, self).__init__(model_id, nn.Sequential(*modules), execution)

    @classmethod
    def build(cls, model_id: AnyStr, de_conv_blocks: List[DeConv2DBlock]) -> Self:
        """
        Create a pure de-convolutional neural network as a convolutional decoder for
        variational auto-encoder or generative adversarial network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param de_conv_blocks: List of de-convolutional blocks
        @type de_conv_blocks: List[DeConvBlock]
        @return: Instance of decoder of type DeConvModel
        @rtype: DeConvModel
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
        logger.info(x, 'Input Conv model')
        x = self.model(x)
        logger.info(x, 'Output Conv model')
        # If a full connected network is appended to the convolutional layers
        if self.ffnn_blocks is not None and len(self.ffnn_blocks) > 0:
            logger.info(x, 'Before width Conv')
            sz = x.shape[0]
            x = DeConvModel.reshape(x, sz)
            logger.info(x, 'After width Conv')
            x = self.dff_model(x)
            logger.info(x, 'Output connected Conv')
        return x

    def list_modules(self, index: int = 0) -> AnyStr:
        modules = [f'{idx + index}: {str(module)}' for idx, module in enumerate(self.get_modules())]
        return '\n'.join(modules)

    def _state_params(self) -> Dict[AnyStr, Any]:
        return {
            "model_id": self.model_id,
            "input_size": self.de_conv_blocks[0].conv_block_config.in_channels,
            "output_size": self.ffnn_blocks[-1].out_features if self.ffnn_blocks is not None else -1,
            "dff_model_input_size": self.ffnn_blocks[0].in_features if self.ffnn_blocks is not None else -1
        }

    @staticmethod
    def is_valid(de_conv_blocks: List[DeConv2DBlock], ffnn_blocks: List[FFNNBlock]) -> bool:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param de_conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type de_conv_blocks: List[ConvBlock]
        @param ffnn_blocks:  List of neural blocks which layout is to be evaluated
        @type ffnn_blocks: List[FFNNBlock]
        """
        try:
            assert de_conv_blocks, 'This convolutional model has not defined neural blocks'
            DeConvModel.__validate(de_conv_blocks)
            if not ffnn_blocks:
                FFNNModel.is_valid(ffnn_blocks)
            return True
        except AssertionError as e:
            logging.error(e)
            return False

    """ ----------------------------   Private helper methods --------------------------- """

    @staticmethod
    def __validate(neural_blocks: List[DeConv2DBlock]):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"
        for index in range(len(neural_blocks) - 1):
            assert neural_blocks[index + 1].in_channels == neural_blocks[index].out_channels, \
                f'Layer {index} input_tensor != layer {index + 1} output'
