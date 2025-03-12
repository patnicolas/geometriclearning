__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.mlp_block import MLPBlock
from dl.block.graph.gconv_block import GConvBlock
from dl.training.neural_training import NeuralTraining
from dl.training.hyper_params import HyperParams
from dl import DLException, GNNException
from typing import List, AnyStr, Optional, Self
import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import logging
logger = logging.getLogger('dl.model.GConvModel')

__all__ = ['GConvModel']


class GConvModel(nn.Module):
    def __init__(self,
                 model_id: AnyStr,
                 gconv_blocks: List[GConvBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        """
        Constructor for this simple Graph convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param gconv_blocks: List of Graph convolutional neural blocks
        @type gconv_blocks: List[ConvBlock]
        @param mlp_blocks: List of Feed-Forward Neural Blocks
        @type mlp_blocks: List[MLPBlock]
        """
        super(GConvModel, self).__init__()

        self.model_id = model_id
        self.gconv_blocks = gconv_blocks
        # Extract the torch modules for the convolutional blocks
        # in the appropriate order
        modules: List[nn.Module] = [module for block in gconv_blocks
                                    for module in block.modules]
        # If fully connected are provided as CNN
        if mlp_blocks is not None:
            self.mlp_blocks = mlp_blocks
            # Flatten the output from the last convolutional layer
            modules.append(nn.Flatten())
            # Extract the relevant modules from the fully connected blocks
            [modules.append(module) for block in mlp_blocks for module in block.modules]
        self.modules = modules

    def forward(self, data: Data) -> torch.Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param data: Graph representation
        @type data: Data
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        # Step 1: Initiate the graph embedding vector
        x = data.x

        # Step 2: Process forward the convolutional layers
        # Create and collect the output of each GNN layer
        aggr = []
        for gconv_block in self.gconv_blocks:
            # Implicit invoke forward method for the block
            x = gconv_block(x, data.edge_index)
            aggr.append(x)

        # Step 3: Concatenate the output of the convolutional layers
        x = torch.cat(aggr, dim=-1)

        # Step 4: Process the fully connected, MLP layers
        for mlp_block in self.mlp_blocks:
            # Invoke the forward method for the MLP block
            x = mlp_block(x)
        return x

    def __str__(self) -> AnyStr:
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules)])

