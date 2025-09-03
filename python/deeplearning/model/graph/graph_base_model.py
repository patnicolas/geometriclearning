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
from typing import List, AnyStr, Optional, Self
from abc import ABC, abstractmethod
# 3rd Party imports
import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# Library imports
from deeplearning.model.neural_model import NeuralModel
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.block.graph.message_passing_block import MessagePassingBlock
from deeplearning.training.gnn_training import GNNTraining
__all__ = ['GraphBaseModel']


class GraphBaseModel(NeuralModel, ABC):

    def __init__(self,
                 model_id: AnyStr,
                 graph_blocks: List[MessagePassingBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        """
        Constructor for this simple generic Graph neural network

        @param model_id: Identifier for this model
        @type model_id: Str
        @param graph_blocks: List of Graph convolutional neural blocks
        @type graph_blocks: List[ConvBlock]
        @param mlp_blocks: List of Feed-Forward Neural Blocks
        @type mlp_blocks: List[MLPBlock]
        """
        assert len(graph_blocks) > 0, f'Number of message passing blocks {graph_blocks} should not be empty'

        self.graph_blocks = graph_blocks
        if mlp_blocks is not None:
            self.mlp_blocks = mlp_blocks
        super(GraphBaseModel, self).__init__(model_id)

    @classmethod
    def build(cls, model_id: AnyStr, gnn_blocks: List[MessagePassingBlock]) -> Self:
        """
        Create a pure graph neural network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param gnn_blocks: List of convolutional blocks
        @type gnn_blocks: List[ConvBlock]
        @return: Instance of decoder of type GCNModel
        @rtype: GraphBaseModel
        """
        return cls(model_id, graph_blocks=gnn_blocks, mlp_blocks=None)

    def get_modules(self) -> List[nn.Module]:
        """
        Extract the ordered list of all the PyTorch modules in this model. The sequence of modules is computed
        the first time it is invoked.
        @return: Ordered list of torch module for this model
        @rtype: List[Module]
        """
        self._register_modules(self.graph_blocks, self.mlp_blocks)
        return list(self.modules_seq.children())

    def __repr__(self) -> str:
        modules = [f'{idx}: {str(module)}' for idx, module in enumerate(self.get_modules())]
        module_repr = '\n'.join(modules)
        return f'\nModules:\n{module_repr}\nBatch size: {self.batch_size}\nWalk length: {self.walk_length}'

    def has_fully_connected(self) -> bool:
        """
        Test if this graph convolutional neural network as a fully connected network
        @return: True if at least one fully connected layer exists, False otherwise
        @rtype: bool
        """
        return len(self.mlp_blocks) > 0

    def forward(self, data: Data) -> torch.Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class.

        @param data: Graph representation
        @type data: Data
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        x = data.x
        edge_index = data.edge_index
        output = []
        for gnn_block in self.graph_blocks:
            x = gnn_block(x, edge_index)
            output.append(x)
        x = torch.cat(output, dim=-1)
        mlp_block = self.mlp_blocks[0]
        # It is assumed that the first module is a neural layer
        linear = mlp_block.modules[0]
        x = linear(x)
        return x

    def reset_parameters(self) -> None:
        """
        Reset the parameters for all the blocks for this model. This method invokes the reset_parameters method for
        each block that in turn reset the parameters for the layer of the block.
        The sequence of modules for this model is computed the first time it is accessed.
        @see NeuralModel._register_modules
        """
        # Register the sequence of torch modules if not defined yet.
        self._register_modules(self.graph_blocks, self.mlp_blocks)

        for graph_sage_block in self.graph_blocks:
            graph_sage_block.reset_parameters()

        # If fully connected perceptron blocks are defined...
        if self.mlp_blocks is not None:
            for mlp_block in self.mlp_blocks:
                mlp_block.reset_parameters()

    def init_weights(self) -> None:
        if self.mlp_blocks is not None:
            for mlp_block in self.mlp_blocks:
                mlp_block.init_weights()

    @abstractmethod
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
        pass

    def load_data_source(self, data_source: Data | Dataset) -> (DataLoader, DataLoader):
        """
        Implement a generic loader for
        @param data_source: Source of data of type Data or Dataset
        @type data_source: Union[Data, Dataset]
        @return: Pair (loader training data, loader test data)
        @rtype: Tuple[DataLoader, DataLoader]
        """
        return self.__load_data(data_source) if isinstance(data_source, Data) \
            else self.__load_dataset(data_source)

    def get_in_features(self) -> int:
        raise NotImplementedError('GCNModel.get_in_features undefined for abstract neural model')

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of input features
        @rtype: int
        """
        raise NotImplementedError('GCNModel.get_out_features undefined for abstract neural model')

    def get_latent_features(self) -> int:
        raise NotImplementedError('GCNModel.get_latent_features undefined for abstract neural model')

    def save(self, extra_params: dict = None):
        raise NotImplementedError('GCNModel.save is an abstract method')

    from torch_geometric.datasets import TUDataset

    @staticmethod
    def convert_dataset_to_data(dataset: TUDataset) -> Data:
        return dataset.data

    """ ----------------------  Private helper methods ------------------  """

    def __load_data(self, data: Data) -> (DataLoader, DataLoader):
        train_loader = GraphSAINTRandomWalkSampler(data=data,
                                                   batch_size=self.batch_size,
                                                   walk_length=self.walk_length,
                                                   num_steps=3,
                                                   is_train=True)
        test_loader = GraphSAINTRandomWalkSampler(data=data,
                                                  batch_size=self.batch_size,
                                                  walk_length=self.walk_length,
                                                  num_steps=3,
                                                  is_train=False)
        return train_loader, test_loader

