__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.mlp_block import MLPBlock
from dl.block.graph.gconv_block import GConvBlock
from dl.model.neural_model import NeuralModel
from typing import List, AnyStr, Optional
import torch
from torch_geometric.data import Data
import torch.nn as nn
import logging
logger = logging.getLogger('dl.model.GConvModel')

__all__ = ['GConvModel']


class GConvModel(NeuralModel):
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
        self.gconv_blocks = gconv_blocks
        # Extract the torch modules for the convolutional blocks
        # in the appropriate order

        gconv_modules: List[nn.Module] = [module for block in gconv_blocks
                                          for module in block.modules_list]
        # If fully connected are provided as CNN
        if mlp_blocks is not None:
            self.mlp_blocks = mlp_blocks
            # Flatten the output from the last convolutional layer
            gconv_modules.append(nn.Flatten())
            # Extract the relevant modules from the fully connected blocks
            mlp_modules: List[nn.Module] = [module for block in mlp_blocks
                                            for module in block.modules_list]
            gconv_modules = gconv_modules + mlp_modules
        super(GConvModel, self).__init__(model_id, nn.Sequential(*gconv_modules))

    def forward(self, data: Data) -> torch.Tensor:
        # Step 1: Initiate the graph embedding vector
        x = data.x

        # Step 2: Process forward the convolutional layers
        # Create and collect the output of each GNN layer
        for gconv_block in self.gconv_blocks:
            # Implicit invoke forward method for the block
            # print(f'Before forward shape {x.shape}')
            x = gconv_block(x, data.edge_index, data.batch)
            # print(f'After forward shape {x.shape}')

        # Step 3: Concatenate the output of the convolutional layers
        # x = torch.cat(aggr, dim=0)

        # Step 4: Process the fully connected, MLP layers
        for mlp_block in self.mlp_blocks:
            # Invoke the forward method for the MLP block
            x = mlp_block(x)
        return x


