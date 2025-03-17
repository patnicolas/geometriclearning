__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import List, AnyStr, Self, Dict
from dl.block.conv import ConvDataType
from dl.model.neural_model import NeuralModel, NeuralBuilder
from dl.block.mlp_block import MLPBlock
import torch.nn as nn
import logging
logger = logging.getLogger('dl.model.FFNNModel')

"""
Class builder for a feed-forward neural network model using feed-forward neural blocks

"""


class MLPModel(NeuralModel):
    def __init__(self, model_id: AnyStr, neural_blocks: List[MLPBlock]) -> None:
        """
        Constructor for the Feed Forward Neural Network model as a set of Neural blocks
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param neural_blocks: List of Neural blocks
        @type neural_blocks:
        """
        assert len(neural_blocks) > 1, 'Cannot create a MLP model without neural blocks'
        self.neural_blocks = neural_blocks

        # Define the sequence of modules from the layout of neural blocks
        modules = [module for block in neural_blocks
                   for module in block.modules]
        super(MLPModel, self).__init__(model_id, nn.Sequential(*modules))

    def transpose(self, output_activation: nn.Module = None) -> Self:
        """
        Generate the inverted neural layout for this Multi-layer perceptron
        @return: This Multi-layer perceptron with an inverted layout
        @rtype: MLPModel
        """
        neural_blocks = [block.transpose(output_activation)
                         for block in self.neural_blocks[::-1]]
        return MLPModel(model_id=f'_{self.model_id}',
                        neural_blocks=neural_blocks)

    def get_in_features(self) -> int:
        """
        Polymorphic method to retrieve the number of input features
        @return: Number of input features
        @rtype: int
        """
        return self.neural_blocks[0].get_in_features()

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of input features
        @rtype: int
        """
        return self.neural_blocks[-1].get_out_features()

    def get_flatten_output_size(self) -> ConvDataType:
        return self.out_features

    def get_latent_features(self) -> int:
        return self.neural_blocks[-1].in_features

    def __str__(self) -> AnyStr:
        return f'\nModel: {self.model_id}\nModules:{self.__repr__()}'

    def __repr__(self) -> AnyStr:
        return f'\n{self.list_modules(0)}'

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')


"""
    Builder for the Multi-layer Perceptron (MLP) model.
    The MLP model is built from a dictionary of configuration parameters for which 
    the keys are predefined. The model is iteratively created by call to method set 
    defined in the base class NeuralBuilder
    The constructor define defaults value for activation (nn.ReLU()) and drop_out (no dropout)
"""


class MLPBuilder(NeuralBuilder):
    keys = ['in_features_list', 'activation', 'drop_out', 'output_activation']

    def __init__(self, model_id: AnyStr) -> None:
        """
        Constructor for the builder using default set of keys (name of configuration
        parameters) and default value for activation module and no dropput
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        """
        super(MLPBuilder, self).__init__(model_id, MLPBuilder.keys)
        # Default configuration parameters that can be overwritten
        self._attributes['activation'] = nn.ReLU()
        self._attributes['drop_out'] = 0.0

    def build(self) -> MLPModel:
        """
        Build the MLP model from a dictionary of configuration parameters in two steps:
        1- Generate the MLP neural blocks from the configuration
        2- Validate the model
        @return: Multi-layer perceptron
        @rtype: MLP model
        """
        # Instantiate the model from the dictionary of
        # Configuration parameters
        mlp_blocks = self.__create_blocks()
        # Validation
        MLPBuilder.validate(mlp_blocks)
        return MLPModel(self._attributes['model_id'], mlp_blocks)

    @staticmethod
    def validate(mlp_blocks: List[MLPBlock]) -> None:
        assert len(mlp_blocks) > 0, "MLP needs at least one layer"

        # Check consistency of number of input and output features
        for index in range(len(mlp_blocks) - 1):
            assert (mlp_blocks[index + 1].get_in_features() ==
                    mlp_blocks[index].get_out_features()), \
                f'Layer {index} input_tensor != layer {index + 1} output'

    """  ---------------  Private Helper Methods ------------- """
    def __create_blocks(self) -> List[MLPBlock]:
        mlp_blocks = []
        in_features_list = self._attributes['in_features_list']
        in_feature = in_features_list[0]

        # Build iteratively the sequence of Feed forward
        # Neural blocks if more than one layer
        for idx in range(1, len(in_features_list)):
            layer = nn.Linear(in_features=in_feature,
                              out_features=in_features_list[idx],
                              bias=False)
            activation_module = self._attributes['output_activation'] \
                if (idx == len(in_features_list) - 1 and
                    self._attributes['output_activation'] is not None) \
                else self._attributes['activation']

            # Build the MLP block
            mlp_block = MLPBlock(
                    block_id=f'{self._attributes["model_id"]}-con-{idx}',
                    layer_module=layer,
                    activation_module=activation_module,
                    dropout_module=nn.Dropout(self._attributes['drop_out']))
            mlp_blocks.append(mlp_block)
            in_feature = in_features_list[idx]
        return mlp_blocks



