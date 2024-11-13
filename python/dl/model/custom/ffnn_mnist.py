__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr
import torch
import torch.nn as nn
from dl.model.custom.base_model import BaseModel
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
import logging
logger = logging.getLogger('dl.model.custom.FFNNMNIST')

__all__ = ['BaseModel', 'FfnnMnist']


class FfnnMnist(BaseModel):
    id = 'FFNN-MNIST'
    default_training_file = 'processed/training.pt'
    default_test_file = 'processed/test.pt'
    num_classes = 10

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
                                            out_features = BaseModel.num_classes,
                                            activation=nn.Softmax(dim=1))

        # Define the model and layout for the Feed Forward Neural Network
        ffnn_model = FFNNModel(FfnnMnist.id, [ffnn_input_block] + ffnn_hidden_blocks + [ffnn_output_block])
        # Invoke base class
        super(FfnnMnist, self).__init__(ffnn_model)

    def _extract_datasets(self, root_path: AnyStr) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Extract the training data and labels and test data and labels for this feed forward neural network
        @param root_path: Root path to MNIST dataset
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        from dl.training.neural_net import NeuralNet
        from torch.nn.functional import one_hot

        _, torch_device = NeuralNet.get_device()

        train_data = torch.load(f'{root_path}/{BaseModel.default_training_file}')
        num_samples = len(train_data[0])
        train_features = train_data[0].reshape(num_samples, -1).float().to(torch_device)
        train_labels = one_hot(train_data[1], num_classes=BaseModel.num_classes).float().to(torch_device)

        test_data = torch.load(f'{root_path}/{BaseModel.default_test_file}')
        num_samples = len(test_data[0])
        test_features = test_data[0].reshape(num_samples, -1).float().to(torch_device)
        test_labels = one_hot(test_data[1], num_classes=BaseModel.num_classes).float().to(torch_device)

        return train_features, train_labels, test_features, test_labels




