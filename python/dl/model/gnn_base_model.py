__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.model.neural_model import NeuralModel
from dl.block.ffnn_block import FFNNBlock
from dl.block.graph.gnn_base_block import GNNBaseBlock
from dl.training.neural_training import NeuralTraining
from dl.training.hyper_params import HyperParams
from dl import DLException, GNNException
from typing import List, AnyStr, Optional, Self
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import logging
logger = logging.getLogger('dl.model.GNNBaseModel')

__all__ = ['GNNBaseModel']


class GNNBaseModel(NeuralModel):

    def __init__(self,
                 model_id: AnyStr,
                 batch_size: int,
                 walk_length: int,
                 gnn_blocks: List[GNNBaseBlock],
                 ffnn_blocks: Optional[List[FFNNBlock]] = None) -> None:
        """
        Constructor for this simple Graph convolutional neural network
        @param model_id: Identifier for this model
        @type model_id: Str
        @param batch_size: Number of “root” nodes (not each batch’s final number of nodes!)
        @type batch_size: int
        @param walk_length: Number of steps to take on each random walk starting from a “root” node
        @type walk_length: int
        @param gnn_blocks: List of Graph convolutional neural blocks
        @type gnn_blocks: List[ConvBlock]
        @param ffnn_blocks: List of Feed-Forward Neural Blocks
        @type ffnn_blocks: List[FFNNBlock]
        """

        assert 0 < batch_size < 8192, f'Batch size {batch_size} if out of range [1, 8192['
        self.batch_size = batch_size
        self.walk_length = walk_length

        modules: List[nn.Module] = [module for block in gnn_blocks for module in block.modules]
        # If fully connected are provided as CNN
        if ffnn_blocks is not None:
            self.ffnn_blocks = ffnn_blocks
            modules.append(nn.Flatten())
            [modules.append(module) for block in ffnn_blocks for module in block.modules]
        super(GNNBaseModel, self).__init__(model_id, nn.Sequential(*modules))

    @classmethod
    def build(cls, model_id: AnyStr, batch_size: int, walk_length: int, gcn_blocks: List[GNNBaseBlock]) -> Self:
        """
        Create a pure convolutional neural network as a graph convolutional encoder for
        variational auto-encoder or generative adversarial network
        @param model_id: Identifier for the model
        @type model_id: AnyStr
        @param batch_size: Number of “root” nodes (not each batch’s final number of nodes!)
        @type batch_size: int
        @param walk_length: Number of steps to take on each random walk starting from a “root” node
        @type walk_length: int
        @param gcn_blocks: List of convolutional blocks
        @type gcn_blocks: List[ConvBlock]
        @return: Instance of decoder of type GCNModel
        @rtype: GNNBaseModel
        """
        return cls(model_id, batch_size=batch_size, walk_length=walk_length, gnn_blocks=gcn_blocks, ffnn_blocks=None)

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

    def do_train(self,
                 data_source: Dataset | Data,
                 hyper_parameters: HyperParams,
                 metric_labels: List[AnyStr]) -> None:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param data_source: Specifically formatted input training data
        @type data_source: Data or Dataset
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metric_labels: List of metrics to be used
        @type metric_labels: List
        """
        try:
            network = NeuralTraining.build(hyper_parameters, metric_labels)
            train_dataset, test_dataset = self.load_data_source(data_source)
            network.train(self.model_id, self.model, train_dataset, test_dataset)
        except GNNException as e:
            logger.error(str(e))
            raise DLException(e)
        except AssertionError as e:
            logger.error(str(e))
            raise DLException(e)

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

    def __load_dataset(self, dataset: Dataset) -> (DataLoader, DataLoader):
        return None

