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

from deeplearning.model.neural_model import NeuralModel
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.block.graph.message_passing_block import MessagePassingBlock
from deeplearning.training.neural_training import NeuralTraining
from deeplearning.training.hyper_params import HyperParams
from metric.built_in_metric import BuiltInMetric
from metric.metric_type import MetricType
from typing import List, AnyStr, Optional, Self, Dict
import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from deeplearning.block.graph import GraphException
import logging

__all__ = ['GNNBaseModel']


class GNNBaseModel(NeuralModel):

    def __init__(self,
                 model_id: AnyStr,
                 gnn_blocks: List[MessagePassingBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        """
        Constructor for this simple Graph convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param gnn_blocks: List of Graph convolutional neural blocks
        @type gnn_blocks: List[ConvBlock]
        @param mlp_blocks: List of Feed-Forward Neural Blocks
        @type mlp_blocks: List[MLPBlock]
        """
        assert len(gnn_blocks) > 0, f'Number of message passing blocks {gnn_blocks} should not be empty'

        self.gnn_blocks = gnn_blocks

        modules: List[nn.Module] = [module for block in gnn_blocks for module in block.modules]
        # If fully connected are provided as CNN
        if mlp_blocks is not None:
            self.ffnn_blocks = mlp_blocks
            # Flatten
            modules.append(nn.Flatten())
            # Generate
            [modules.append(module) for block in mlp_blocks for module in block.modules]
        super(GNNBaseModel, self).__init__(model_id, nn.Sequential(*modules))

    @classmethod
    def build(cls, model_id: AnyStr, gnn_blocks: List[MessagePassingBlock]) -> Self:
        """
        Create a pure graph neural network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param gnn_blocks: List of convolutional blocks
        @type gnn_blocks: List[ConvBlock]
        @return: Instance of decoder of type GCNModel
        @rtype: GNNBaseModel
        """
        return cls(model_id, gnn_blocks=gnn_blocks, mlp_blocks=None)

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
        return len(self.ffnn_blocks) > 0

    def forward(self, data: Data) -> torch.Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param data: Graph representation
        @type data: Data
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        x = data.x
        edge_index = data.edge_index
        output = []
        for gnn_block in self.gnn_blocks:
            x = gnn_block(x, edge_index)
            output.append(x)
        x = torch.cat(output, dim=-1)
        ffnn = self.ffnn_blocks[0]
        linear = ffnn.modules[0]
        x = linear(x)
        return x

    def do_train(self,
                 hyper_parameters: HyperParams,
                 metrics_list: List[MetricType],
                 data_source: Data | Dataset) -> None:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metrics_list: List of performance metrics
        @type metrics_list: List
        @param data_source: Data source
        @type data_source: Union[Data, Dataset]
        """
        try:
            metrics_attributes = {metric_type: BuiltInMetric(metric_type) for metric_type in metrics_list}
            network = NeuralTraining(hyper_parameters, metrics_attributes)
            train_dataset, test_dataset = self.load_data_source(data_source)
            network.train(self.model_id, self.modules_seq, train_dataset, test_dataset)
        except AssertionError as e:
            logging.error(str(e))
            raise GraphException(e)

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

