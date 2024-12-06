__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import List, AnyStr, Self
from dl.model.neural_model import NeuralModel
from dl.block.ffnn_block import FFNNBlock
import torch
import torch.nn as nn
import logging
logger = logging.getLogger('dl.model.FFNNModel')

"""
Class builder for a feed-forward neural network model using feed-forward neural blocks

"""


class FFNNModel(NeuralModel):
    def __init__(self, model_id: AnyStr, neural_blocks: List[FFNNBlock]) -> None:
        """
        Constructor for the Feed Forward Neural Network model as a set of Neural blocks
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param neural_blocks: List of Neural blocks
        @type neural_blocks:
        """
        FFNNModel.is_valid(neural_blocks)

        self.neural_blocks = neural_blocks
        # Record the number of input and output features from the first and last neural block respectively
        self.in_features = neural_blocks[0].in_features
        self.out_features = neural_blocks[-1].out_features

        # Define the sequence of modules from the layout
        modules = [module for block in neural_blocks for module in block.modules]
        super(FFNNModel, self).__init__(model_id, torch.nn.Sequential(*modules))

    def transpose(self, extra: nn.Module = None) -> Self:
        """
        Generate the inverted neural layout for this feed forward neural network
        @return: This feed-forward neural network with an inverted layout
        @rtype: FFNNModel
        """
        neural_blocks: list[FFNNBlock] = [block.transpose() for block in self.neural_blocks[::-1]]
        return FFNNModel(model_id=f'_{self.model_id}', neural_blocks=neural_blocks)

    def get_in_features(self) -> int:
        """
        Polymorphic method to retrieve the number of input features
        @return: Number of input features
        @rtype: int
        """
        return self.in_features

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of input features
        @rtype: int
        """
        return self.out_features

    def get_latent_features(self) -> int:
        return self.neural_blocks[-1].in_features

    def __repr__(self) -> AnyStr:
        blocks_str = '\n'.join([f'{idx+1}:   {repr(block)}' for idx, block in enumerate(self.neural_blocks)])
        return f'\n      Id: {self.model_id}\n{blocks_str}'

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')

    @staticmethod
    def is_valid(neural_blocks: List[FFNNBlock]) -> bool:
        try:
            assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"
            for index in range(len(neural_blocks) - 1):
                assert neural_blocks[index + 1].in_features == neural_blocks[index].out_features, \
                    f'Layer {index} input_tensor != layer {index+1} output'
            return True
        except AssertionError as e:
            logging.error(str(e))
            return False
