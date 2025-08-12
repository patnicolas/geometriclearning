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
from torch.utils.data import DataLoader
# Library imports
from deeplearning.model.neural_model import NeuralBuilder
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.block.graph.graph_sage_block import GraphSAGEBlock
from deeplearning.model.graph.graph_base_model import GraphBaseModel
from deeplearning.training.gnn_training import GNNTraining
__all__ = ['GraphSAGEModel', 'GraphSAGEBuilder']


class GraphSAGEModel(GraphBaseModel):

    def __init__(self,
                 model_id: AnyStr,
                 graph_SAGE_blocks: List[GraphSAGEBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        super(GraphSAGEModel, self).__init__(model_id, graph_SAGE_blocks, mlp_blocks)

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

        # Step 2: Process forward the SAGE layers
        # Create and collect the output of each GNN layer
        for graph_SAGE_block in self.graph_blocks:
            # Implicit invoke forward method for the block
            x = graph_SAGE_block(x, edge_index, data.batch)
        # Step 4: Process the fully connected, MLP layers
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)  # Invoke the forward method for the MLP block
        return x

    def train_model(self, gnn_training: GNNTraining, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Training and evaluation of models using Graph Neural Training and train loader for training and evaluation data

        @param gnn_training: Wrapper class for training Graph Neural Network
        @type gnn_training:  GNNTraining
        @param train_loader: Loader for the training data set
        @type train_loader: torch.utils.data.DataLoader
        @param val_loader:   Loader for the validation data set
        @type val_loader:  torch.utils.data.DataLoader
        """
        gnn_training.train(neural_model=self,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           val_enabled=True)


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
        1- Generate the graph SAGE neural block from the configuration parameters
        2- Generate the MLP neural blocks from the configuration if defined
        3- Validate the model
        @return: Graph SAGE model instance
        @rtype: GraphSAGEModel
        """
        graph_SAGE_blocks_attribute = self.model_attributes['graph_SAGE_blocks']
        mlp_blocks_attribute = self.model_attributes['mlp_blocks']
        graph_SAGE_blocks = [GraphSAGEBlock.build(graph_SAGE_block_attribute)
                             for graph_SAGE_block_attribute in graph_SAGE_blocks_attribute]
        mlp_blocks = [MLPBlock.build(mlp_block_attribute) for mlp_block_attribute in mlp_blocks_attribute]
        return GraphSAGEModel(self.model_attributes['model_id'], graph_SAGE_blocks, mlp_blocks)
