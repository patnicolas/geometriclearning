__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.loader import (NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler,
                                    GraphSAINTNodeSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler,
                                    ClusterData, ClusterLoader)
from typing import Dict, AnyStr, Any
from dataset import DatasetException
__all__ = ['GraphDataLoader']


"""
Universal wrapper for the various Graph Data Loaders to generate the training and evaluation data loader to
be directly used for training and evaluation of graph neural network models.
The steps are
1- Specify the dictionary of attributes (arguments) for a given data loader (i.e. num_neighbors, batch_size and replace) 
   for NeighborsLoader
2- Specify the data to be loaded (including the train and valuation masks)
3- Use mask and/or Shuffle attribute to build the training and test loaders
4- Specify the number of workers if > 1

The graph data loader currently supported are:
NeighborLoader
RandomNodeLoader
GraphSAINTRandomWalkSampler
GraphSAINTNodeSampler
GraphSAINTEdgeSampler
ShaDowKHopSampler
ClusterLoader
"""


class GraphDataLoader(object):
    def __init__(self, loader_attributes: Dict[AnyStr, Any], data: Data) -> None:
        """
        Constructor for the Generic Graph Data Loader
        @param loader_attributes: Map for attributes for a given Data loader
        @type loader_attributes: Dict[AnyStr, Any]
        @param data: Data with features, labels, and masks for training and evaluation
        @type data: torch_geometric.data.Data
        """
        GraphDataLoader.__validate(loader_attributes)
        self.data = data
        self.attributes_map = loader_attributes

    def __call__(self) -> (DataLoader, DataLoader):
        """
        Generate the data loader for both training and evaluation set
        @return: A pair of training loader and evaluation loader
        @rtype: Tuple[DataLoader, DataLoader]
        """
        match self.attributes_map['id']:
            case 'NeighborLoader':
                return self.__neighbors_loader()
            case 'RandomNodeLoader':
                return self.__random_node_loader()
            case 'GraphSAINTNodeSampler':
                return self.__graph_saint_node_sampler()
            case 'GraphSAINTEdgeSampler':
                return self.__graph_saint_edge_sampler()
            case 'ShaDowKHopSampler':
                return self.__shadow_khop_sampler()
            case 'GraphSAINTRandomWalkSampler':
                return self.__graph_saint_random_walk()
            case 'ClusterLoader':
                return self.__cluster_loader()
            case _:
                raise DatasetException(f'Graph data loader {self.attributes_map["id"]} is not supported')

    def __str__(self) -> AnyStr:
        return f'\nAttributes: {str(self.attributes_map)}\nData: {self.data}'

    def __len__(self) -> int:
        return len(self.data.x)

    """ ------------------------ Private Helper Methods ------------------------ """

    def __random_node_loader(self) -> (DataLoader, DataLoader):
        num_parts = self.attributes_map['num_parts']
        train_loader = RandomNodeLoader(self.data, num_parts=num_parts, shuffle=True, mask=self.data.train_mask)
        eval_loader = RandomNodeLoader(self.data, num_parts=num_parts, shuffle=False, mask=self.data.val_mask)
        return train_loader, eval_loader

    def __neighbors_loader(self) -> (DataLoader, DataLoader):
        num_neighbors = self.attributes_map['num_neighbors']
        batch_size = self.attributes_map['batch_size']
        replace = self.attributes_map['replace']
        train_loader = NeighborLoader(self.data,
                                      num_neighbors=num_neighbors,
                                      batch_size=batch_size,
                                      replace=replace,
                                      shuffle=True,
                                      input_nodes=self.data.train_mask)
        eval_loader = NeighborLoader(self.data,
                                     num_neighbors=num_neighbors,
                                     batch_size=batch_size,
                                     replace=replace,
                                     shuffle=False,
                                     input_nodes=self.data.val_mask)
        return train_loader, eval_loader

    def __graph_saint_node_sampler(self) -> (DataLoader, DataLoader):
        batch_size = self.attributes_map['batch_size']
        num_steps = self.attributes_map['num_steps']
        sample_coverage = self.attributes_map['sample_coverage']
        train_loader = GraphSAINTNodeSampler(data=self.data,
                                             batch_size=batch_size,
                                             num_steps=num_steps,
                                             sample_coverage=sample_coverage,
                                             shuffle=True)
        eval_loader = GraphSAINTNodeSampler(data=self.data,
                                            batch_size=batch_size,
                                            num_steps=num_steps,
                                            sample_coverage=sample_coverage,
                                            shuffle=False)
        return train_loader, eval_loader

    def __graph_saint_edge_sampler(self) -> (DataLoader, DataLoader):
        batch_size = self.attributes_map['batch_size']
        num_steps = self.attributes_map['num_steps']
        sample_coverage = self.attributes_map['sample_coverage']
        train_loader = GraphSAINTEdgeSampler(data=self.data,
                                             batch_size=batch_size,
                                             num_steps=num_steps,
                                             sample_coverage=sample_coverage,
                                             shuffle=True)
        eval_loader = GraphSAINTEdgeSampler(data=self.data,
                                            batch_size=batch_size,
                                            num_steps=num_steps,
                                            sample_coverage=sample_coverage,
                                            shuffle=False)
        return train_loader, eval_loader

    def __graph_saint_random_walk(self) -> (DataLoader, DataLoader):
        walk_length = self.attributes_map['walk_length']
        batch_size = self.attributes_map['batch_size']
        num_steps = self.attributes_map['num_steps']
        sample_coverage = self.attributes_map['sample_coverage']
        train_loader = GraphSAINTRandomWalkSampler(data=self.data,
                                                   batch_size=batch_size,
                                                   walk_length=walk_length,
                                                   num_steps=num_steps,
                                                   sample_coverage=sample_coverage,
                                                   shuffle=True)
        eval_loader = GraphSAINTRandomWalkSampler(data=self.data,
                                                  batch_size=batch_size,
                                                  walk_length=walk_length,
                                                  num_steps=num_steps,
                                                  sample_coverage=sample_coverage,
                                                  shuffle=False)
        return train_loader, eval_loader

    def __shadow_khop_sampler(self) -> (DataLoader, DataLoader):
        depth = self.attributes_map['depth']
        num_neighbors = self.attributes_map['num_neighbors']
        node_idx = self.attributes_map['node_idx']
        replace = self.attributes_map['replace']
        train_loader = ShaDowKHopSampler(data=self.data,
                                         depth=depth,
                                         num_neighbors=num_neighbors,
                                         node_idx=node_idx,
                                         replace=replace,
                                         shuffle=True)
        eval_loader = ShaDowKHopSampler(data=self.data,
                                        depth=depth,
                                        num_neighbors=num_neighbors,
                                        node_idx=node_idx,
                                        replace=replace,
                                        shuffle=False)
        return train_loader, eval_loader

    def __cluster_loader(self) -> (DataLoader, DataLoader):
        num_parts = self.attributes_map['num_parts']
        recursive = self.attributes_map['recursive']
        batch_size = self.attributes_map['batch_size']
        keep_inter_cluster_edges = self.attributes_map['keep_inter_cluster_edges']
        cluster_data = ClusterData(data=self.data,
                                   num_parts=num_parts,
                                   recursive=recursive,
                                   keep_inter_cluster_edges=keep_inter_cluster_edges)
        train_loader = ClusterLoader(data=cluster_data,
                                     batch_size=batch_size,
                                     shuffle=True)
        eval_loader = ClusterLoader(data=cluster_data,
                                    batch_size=batch_size,
                                    shuffle=False)
        return train_loader, eval_loader

    @staticmethod
    def __validate(attributes_map: Dict[AnyStr, Any]) -> None:
        is_valid = False
        if attributes_map is not None and 'id' in attributes_map:
            match attributes_map['id']:
                case 'NeighborLoader':
                    is_valid = ('num_neighbors' in attributes_map and
                                'batch_size' in attributes_map and
                                'replace' in attributes_map)

                case 'RandomNodeLoader':
                    is_valid = 'num_parts' in attributes_map

                case 'GraphSAINTNodeSampler' | 'GraphSAINTEdgeSampler':
                    is_valid = ('batch_size' in attributes_map and
                                'num_steps' in attributes_map and
                                'sample_coverage' in attributes_map)

                case 'GraphSAINTRandomWalkSampler':
                    is_valid = ('walk_length' in attributes_map and
                                'batch_size' in attributes_map and
                                'num_steps' in attributes_map and
                                'sample_coverage' in attributes_map)

                case 'ShaDowKHopSampler':
                    is_valid = ('depth' in attributes_map and
                                'num_neighbors' in attributes_map and
                                'node_idx' in attributes_map and
                                'replace' in attributes_map)

                case 'ClusterLoader':
                    is_valid = ('num_parts' in attributes_map and
                                'recursive' in attributes_map and
                                'batch_size' in attributes_map and
                                'keep_inter_cluster_edges' in attributes_map)
        if not is_valid:
            raise DatasetException(GraphDataLoader.__attrs_map_def)

    __attrs_map_def: AnyStr =('Attributes map:\nNeighborLoader: num_neighbors, batch_size, replace'
                              '\nRandomNodeLoader: num_parts\nGraphSAINTRandomWalkSampler: walk_length, batch_size'
                              'num_steps, sample_coverage\nGraphSAINTNodeSampler: alk_length, batch_size, num_steps'
                              '\nGraphSAINTNodeSampler: walk_length, batch_size um_steps\nShaDowKHopSampler: depth'
                              'num_neighbors node_idx\nClusterLoader: num_parts, recursive batch_size'
                              'keep_inter_cluster_edges')






