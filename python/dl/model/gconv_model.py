__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dl.block.mlp_block import MLPBlock
from dl.block.graph.gconv_block import GConvBlock
from dl.model.neural_model import NeuralModel
from typing import List, AnyStr, Optional, Any, Dict, Self
import torch
from torch_geometric.data import Data
import torch.nn as nn
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

    @classmethod
    def build(cls, model_attributes: Dict[AnyStr, Any]) -> Self:
        gconv_blocks_attribute = model_attributes['gconv_blocks']
        mlp_blocks_attribute = model_attributes['mlp_blocks']
        gconv_blocks = [GConvBlock.build(gconv_block_attribute) for gconv_block_attribute in gconv_blocks_attribute]
        mlp_blocks = [MLPBlock.build(mlp_block_attribute) for mlp_block_attribute in mlp_blocks_attribute]
        return cls(model_attributes['model_id'], gconv_blocks, mlp_blocks)

    def forward(self, data: Data) -> torch.Tensor:
        # Step 1: Initiate the graph embedding vector
        x = data.x

        # Step 2: Process forward the convolutional layers
        # Create and collect the output of each GNN layer
        for gconv_block in self.gconv_blocks:
            # Implicit invoke forward method for the block
            # logging.info(f'Before forward shape {x.shape}')
            x = gconv_block(x, data.edge_index, data.batch)
            # logging.info(f'After forward shape {x.shape}')

        # Step 4: Process the fully connected, MLP layers
        for mlp_block in self.mlp_blocks:
            # Invoke the forward method for the MLP block
            x = mlp_block(x)
        return x


