__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr
import torch
import torch.nn as nn
from dl.model.custom.base_mnist import BaseMNIST
from dl.block.ffnnblock import FFNNBlock
from dl.model.ffnnmodel import FFNNModel
import logging
logger = logging.getLogger('dl.model.custom.FFNNMNIST')

__all__ = ['BaseMNIST', 'FFNNMNIST']



class FFNNMNIST(BaseMNIST):
    id = 'Feed Forward Neural Network MNIST'

    def __init__(self, input_size: int, features: List[int]) -> None:
        """
        Constructor for the feed forward neural network dedicated to MNIST data
        @param input_size: Input size (height and width) of the image (digits)
        @type input_size: int
        @param features: List of number of channels for the hidden and output layer
        @type features: List
        """
        assert len(features) > 0, f'Number of input features are undefined'

        # Input layer
        ffnn_input_block = FFNNBlock.build(
            block_id='input',
            in_features=input_size,
            out_features=features[0],
            activation=nn.ReLU())

        # Hidden layers if any
        ffnn_hidden_blocks = [FFNNBlock.build(block_id=f'hidden_{idx+1}',
                                              in_features=features[idx],
                                              out_features=features[idx+1],
                                              activation=nn.ReLU()) for idx in range(len(features[:-1]))]
        # Output layer
        ffnn_output_block = FFNNBlock.build(block_id='output',
                                            in_features=features[-1],
                                            out_features = BaseMNIST.num_classes,
                                            activation=nn.Softmax(dim=1))

        # Define the model and layout for the Feed Forward Neural Network
        ffnn_model = FFNNModel(FFNNMNIST.id, [ffnn_input_block] + ffnn_hidden_blocks + [ffnn_output_block])
        # Invoke base class
        super(FFNNMNIST, self).__init__(ffnn_model)

    def _process_data(self, root_path: AnyStr) ->(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        from dl.training.neuralnet import NeuralNet

        _, torch_device = NeuralNet.get_device()

        train_data = torch.load(f'{root_path}/{BaseMNIST.default_training_file}')
        num_samples = len(train_data[0])
        train_features = train_data[0].reshape(num_samples, -1).float().to(torch_device)
        train_labels = torch.nn.functional.one_hot(train_data[1], num_classes=10).float().to(torch_device)

        test_data = torch.load(f'{root_path}/{BaseMNIST.default_test_file}')
        num_samples = len(test_data[0])
        test_features = test_data[0].reshape(num_samples, -1).float().to(torch_device)
        test_labels = torch.nn.functional.one_hot(test_data[1], num_classes=10).float().to(torch_device)

        return train_features, train_labels, test_features, test_labels




