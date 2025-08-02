__author__ = "Patrick R. Nicolas"
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

from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.block.graph.graph_conv_block import GraphConvBlock
from deeplearning.model.neural_model import NeuralModel
from typing import List, AnyStr, Optional, Any, Dict, Self
import torch
from torch_geometric.data import Data
import torch.nn as nn
__all__ = ['GraphSAGEModel']


class GraphSAGEModel(NeuralModel):

    def __init__(self,
                 model_id: AnyStr,
                 graph_sage_blocks: List[GraphConvBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        assert len(graph_sage_blocks) > 0, f'Number of graph SAGE block {graph_sage_blocks} should not be empty'

        self.graph_sage_blocks = graph_sage_blocks
        # Extract the torch modules for the SAGE blocks in the appropriate order
        graph_sage_modules: List[nn.Module] = [module for block in graph_sage_blocks
                                               for module in block.modules_list]
        # If fully connected are provided as CNN
        if mlp_blocks is not None:
            self.mlp_blocks = mlp_blocks
            # Flatten the output from the last convolutional layer
            graph_sage_modules.append(nn.Flatten())
            # Extract the relevant modules from the fully connected blocks
            mlp_modules: List[nn.Module] = [module for block in mlp_blocks
                                            for module in block.modules_list]
            graph_sage_modules = graph_sage_modules + mlp_modules
        super(GraphSAGEModel, self).__init__(model_id, nn.Sequential(*graph_sage_modules))

