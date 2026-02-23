__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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
from typing import AnyStr, Optional, Generic, TypeVar, Dict, Any
import logging
# 3rd Party imports
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
# Library imports
from deeplearning.model.graph.graph_base_model import GraphBaseModel
from deeplearning.block.graph.graph_attention_block import GraphAttentionBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.training.gnn_training import GNNTraining
from deeplearning.model.neural_model import NeuralBuilder
import python
__all__ = ['GraphAttentionModel']

GATL = TypeVar("GATL")

class GraphAttentionModel(GraphBaseModel, Generic[GATL]):
    """
        A Graph Attention Network may require one output multi-layer perceptron for classification purpose using
        SoftMax activation. We do not restrict a model from have multiple linear layers for output.
    """
    def __init__(self,
                 model_id: AnyStr,
                 graph_attention_blocks: frozenset[GraphAttentionBlock[GATL]],
                 mlp_blocks: Optional[frozenset[MLPBlock]] = None) -> None:
        """
           Constructor for this simple Graph Attention neural network

           @param model_id: Identifier for this model
           @type model_id: Str
           @param graph_conv_blocks: List of Graph Attention neural blocks
           @type graph_conv_blocks: List[GraphAttentionBlock]
           @param mlp_blocks: List of Feed-Forward Neural Blocks
           @type mlp_blocks: List[MLPBlock]
        """
        super(GraphAttentionModel, self).__init__(model_id, graph_attention_blocks, mlp_blocks)

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
        for graph_attention_block in self.graph_attension_blocks:
            # Implicit invoke forward method for the block
            logging.debug(f'Before Conv shape {x.shape} & index {edge_index}')
            x = graph_attention_block(x, edge_index, data.batch)
            logging.debug(f'After Conv shape {x.shape} & index {edge_index}')

        # Step 3: Process the fully connected, MLP layers
        for mlp_block in self.mlp_blocks:
            # Invoke the forward method for the MLP block
            x = mlp_block(x)
        return x

    def train_model(self, training: GNNTraining, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Training and evaluation of models using Graph Neural Training and train loader for training and evaluation data
        @param training: Wrapper class for training Graph Neural Network
        @type training:  GNNTraining
        @param train_loader: Loader for the training data set
        @type train_loader: torch.utils.data.DataLoader
        @param val_loader: Loader for the validation data set
        @type val_loader: torch.utils.data.DataLoader
        """
        training.train(neural_model=self, train_loader=train_loader, val_loader=val_loader)


class GraphAttentionBuilder(NeuralBuilder):
    """
    Builder for a Graph Attention Model
    The graph attention model is built from a dictionary of configuration parameters
    for which  the keys are predefined. The model is iteratively created by call to method set
    defined in the base class NeuralBuilder

    The constructor define defaults value for activation (nn.ReLU()), stride, padding,
    enabling batch normalization and drop_out (no dropout).
    """
    def __init__(self, model_attributes: Dict[AnyStr, Any]) -> None:
        """
        Constructor for Graph Attention Model using default
        set of keys (name of configuration parameters) and default value for activation
        module, stride, padding, enabling batch normalization and no dropout
        @param model_attributes: Dictionary of model attributes
        @type model_attributes: Dict[AnyStr, Any]
        """
        super(GraphAttentionBuilder, self).__init__(model_attributes)

    def build(self) -> GraphAttentionModel:
        """
        Build Graph attention Model from a dictionary of configuration
        parameters in three steps:
        1- Generate the attention neural block from the configuration parameters
        2- Generate the MLP neural blocks from the configuration if defined
        3- Validate the model
        @return: 2-dimensional convolutional model instance
        @rtype: Conv2dModel
        """
        graph_conv_blocks_attribute = self.model_attributes['graph_conv_blocks']
        mlp_blocks_attribute = self.model_attributes['mlp_blocks']
        graph_attention_blocks = frozenset([GraphAttentionBlock.build(graph_conv_block_attribute)
                                            for graph_conv_block_attribute in graph_conv_blocks_attribute])
        mlp_blocks = frozenset([MLPBlock.build(mlp_block_attribute) for mlp_block_attribute in mlp_blocks_attribute])
        return GraphAttentionModel(self.model_attributes['model_id'], graph_attention_blocks, mlp_blocks)
