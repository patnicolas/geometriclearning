__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr, Self
from python.dl.model.neuralmodel import NeuralModel
from python.dl.block.ffnnblock import FFNNBlock
import torch


class FFNNModel(NeuralModel):

    def __init__(self, model_id: AnyStr, neural_blocks: List[FFNNBlock]):
        """
        Constructor for the Feed Forward Neural Network model as a set of Neural blocks
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param neural_blocks: List of Neural blocks
        @type neural_blocks:
        """
        FFNNModel.__validate(neural_blocks)
        self.neural_blocks = neural_blocks

        # Define the sequence of modules from the layout
        self.in_features = neural_blocks[0].in_features
        self.out_features = neural_blocks[-1].out_features
        modules = [module for layer in neural_blocks for module in layer.modules]
        super(FFNNModel, self).__init__(model_id, torch.nn.Sequential(*modules))

    def invert(self) -> Self:
        """
        Generate the inverted neural layout for this feed forward neural network
        @return: This feed-forwqrd neural network with an inverted layout
        @rtype: FFNNModel
        """
        neural_blocks = [block.invert() for block in self.neural_blocks[::-1]]
        return FFNNModel(f'_{self.model_id}', neural_blocks)

    def get_in_features(self) -> int:
        return self.in_features

    def get_out_features(self) -> int:
        return self.out_features

    def __repr__(self) -> AnyStr:
        blocks_str = '\n'.join([f'   {repr(block)}' for block in self.neural_blocks])
        return f'\n      Id: {self.model_id}\n{blocks_str}'

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')

    @staticmethod
    def __validate(neural_blocks: List[FFNNBlock]):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"
        for index in range(len(neural_blocks) - 1):
            assert neural_blocks[index + 1].in_features == neural_blocks[index].out_features, \
                f'Layer {index} input_tensor != layer {index+1} output'
