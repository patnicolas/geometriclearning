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

# Standard Library imports
from typing import List, AnyStr, Optional, Any, Dict
# 3rd Party imports
from torch_geometric.data import Data
import torch
import torch.nn as nn
# Library imports
from deeplearning.model.neural_model import NeuralModel, NeuralBuilder
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.block.graph.graph_sage_block import GraphSAGEBlock
__all__ = ['GraphSAGEModel', 'GraphSAGEBuilder']


class GraphSAGEModel(NeuralModel):

    def __init__(self,
                 model_id: AnyStr,
                 graph_SAGE_blocks: List[GraphSAGEBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        assert len(graph_SAGE_blocks) > 0, f'Number of graph SAGE block {graph_SAGE_blocks} should not be empty'

        self.graph_SAGE_blocks = graph_SAGE_blocks
        # Extract the torch modules for the SAGE blocks in the appropriate order
        graph_sage_modules: List[nn.Module] = [module for block in graph_SAGE_blocks
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

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward propagation of data across various Graph Convolutional layers and optionally multi-perceptron layers
        @param data: Graph data
        @type data: Data
        @return: values of output features
        @rtype: torch Tensor
        """
        # Step 1: Initiate the graph embedding vector
        x = data.x
        edge_index = data.edge_index

        # Step 2: Process forward the convolutional layers
        # Create and collect the output of each GNN layer
        for graph_SAGE_block in self.graph_SAGE_blocks:
            # Implicit invoke forward method for the block
            x = graph_SAGE_block(x, edge_index, data.batch)

        # Step 4: Process the fully connected, MLP layers
        for mlp_block in self.mlp_blocks:
            # Invoke the forward method for the MLP block
            x = mlp_block(x)
        return x


class GraphSAGEBuilder(NeuralBuilder):
    """
    Builder for a Graph SAGE Model
    The graph SAGE model is built from a dictionary of configuration parameters
    for which  the keys are predefined. The model is iteratively created by call to method set
    defined in the base class NeuralBuilder

    The constructor define defaults value for activation (nn.ReLU()), stride, padding,
    enabling batch normalization and drop_out (no dropout).

    Reference: https://patricknicolas.substack.com/p/modular-deep-learning-models-with
    """
    def __init__(self, model_attributes: Dict[AnyStr, Any]) -> None:
        """
        Constructor for Graph SAGE Model using default
        set of keys (name of configuration parameters) and default value for activation
        module, stride, padding, enabling batch normalization and no dropout
        @param model_attributes: Dictionary of model attributes
        @type model_attributes: Dict[AnyStr, Any]
        """
        super(GraphSAGEBuilder, self).__init__(model_attributes)

    def build(self) -> GraphSAGEModel:
        """
        Build Graph SAGE Model from a dictionary of configuration
        parameters in three steps:
        1- Generate the convolutional neural block from the configuration parameters
        2- Generate the MLP neural blocks from the configuration if defined
        3- Validate the model
        @return: Graph SAGE model instance
        @rtype: Conv2dModel
        """
        graph_SAGE_blocks_attribute = self.model_attributes['graph_SAGE_blocks']
        mlp_blocks_attribute = self.model_attributes['mlp_blocks']
        graph_SAGE_blocks = [GraphSAGEBlock.build(graph_SAGE_block_attribute)
                             for graph_SAGE_block_attribute in graph_SAGE_blocks_attribute]
        mlp_blocks = [MLPBlock.build(mlp_block_attribute) for mlp_block_attribute in mlp_blocks_attribute]
        return GraphSAGEModel(self.model_attributes['model_id'], graph_SAGE_blocks, mlp_blocks)
