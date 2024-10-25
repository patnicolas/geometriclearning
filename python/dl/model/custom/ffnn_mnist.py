__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr
import torch.nn as nn
from dl.model.custom.base_mnist import BaseMNIST
from dl.block.ffnnblock import FFNNBlock
from dl.model.ffnnmodel import FFNNModel
import logging
logger = logging.getLogger('dl.model.custom.FFNNMNIST')



class FFNNMNIST(BaseMNIST):
    id = 'Feed Forward Neural Network MNIST'

    def __init__(self, input_size: int, in_channels: List[int]) -> None:
        """
        Constructor for the feed forward neural network dedicated to MNIST data
        @param input_size: Input size (height and width) of the image (digits)
        @type input_size: int
        @param in_channels: List of number of channels for the hidden and output layer
        @type in_channels: List
        """
        num_classes = 10
        # Input layer
        ffnn_input_block = FFNNBlock.build(
            block_id='input',
            in_features=input_size * input_size,
            out_features=in_channels[0],
            activation=nn.ReLU())

        # Hidden layers if any
        ffnn_hidden_blocks = [FFNNBlock.build(block_id=f'hidden_{idx+1}',
                                              in_features=in_channels[idx+1],
                                              out_features=in_channels[idx+1],
                                              activation=nn.ReLU()) for idx in range(len(in_channels[:-1]))]
        # Output layer
        ffnn_output_block = FFNNBlock.build(block_id='output',
                                            in_features=in_channels[-1],
                                            out_features = num_classes,
                                            activation=nn.Softmax(dim=1))

        # Define the model and layout for the Feed Forward Neural Network
        ffnn_model = FFNNModel(FFNNMNIST.id, [ffnn_input_block] + ffnn_hidden_blocks + [ffnn_output_block])
        super(FFNNMNIST, self).__init__(ffnn_model)

    def __repr__(self) -> AnyStr:
        return repr(self.model)



