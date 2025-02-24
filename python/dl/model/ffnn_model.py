__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import List, AnyStr, Self
from dl import ConvDataType
from dl.model.neural_model import NeuralModel
from dl.block.ffnn_block import FFNNBlock
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
        # Record the number of input and output features from
        # the first and last neural block respectively
        self.in_features = neural_blocks[0].in_features
        self.out_features = neural_blocks[-1].out_features

        # Define the sequence of modules from the layout of neural blocks
        modules = [module for block in neural_blocks for module in block.modules]
        super(FFNNModel, self).__init__(model_id, nn.Sequential(*modules))

    @classmethod
    def build(cls,
              model_id: AnyStr,
              in_features: List[int],
              activation: nn.Module,
              drop_out: float,
              output_activation: nn.Module = None) -> Self:
        ffnn_blocks = FFNNModel.create_ffnn_blocks(
            model_id,
            in_features,
            activation,
            drop_out,
            output_activation
        )
        return cls(model_id, ffnn_blocks)

    @staticmethod
    def create_ffnn_blocks(
            model_id: AnyStr,
            in_features: List[int],
            activation: nn.Module,
            drop_out: float = 0.2,
            output_activation: nn.Module = None) -> List[FFNNBlock]:
        ffnn_blocks = []
        in_feature = in_features[0]
        for idx in range(1, len(in_features)):
            layer = nn.Linear(in_features=in_feature,
                              out_features=in_features[idx],
                              bias=False)
            activation_module = output_activation \
                if idx == len(in_features) - 1 and activation_module is not None \
                else activation
            ffnn_block = FFNNBlock.build(
                block_id=f'{model_id}-con-{idx}',
                layer=layer,
                activation=activation_module,
                drop_out=drop_out)
            ffnn_blocks.append(ffnn_block)
            in_feature = in_features[idx]
        return ffnn_blocks

    def transpose(self, output_activation: nn.Module = None) -> Self:
        """
        Generate the inverted neural layout for this feed forward neural network
        @return: This feed-forward neural network with an inverted layout
        @rtype: FFNNModel
        """
        neural_blocks = [block.transpose(output_activation) for block in self.neural_blocks[::-1]]
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

    def get_flatten_output_size(self) -> ConvDataType:
        return self.out_features

    def get_latent_features(self) -> int:
        return self.neural_blocks[-1].in_features

    def __str__(self) -> AnyStr:
        return f'\nModel: {self.model_id}\nModules:\n{self.list_modules(0)}'

    def __repr__(self) -> AnyStr:
        return f'\n{self.list_modules(0)}'

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
