__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler
from typing import Dict, AnyStr, Any, Self
from dataset import DatasetException
__all__ = ['GraphDataLoader']


class GraphDataLoader(object):
    def __init__(self, attributes_map: Dict[AnyStr, Any], data: Data) -> None:
        GraphDataLoader.__validate(attributes_map)
        self.data = data
        self.attributes_map = attributes_map

    @classmethod
    def build(cls, attributes_map: Dict[AnyStr, Any], dataset: Dataset) -> Self:
        data: Data = dataset[0]
        return cls(attributes_map, data)

    def __call__(self) -> DataLoader:
        match self.attributes_map['id']:
            case 'NeighborLoader':
                return self.__neighbors_loader()
            case 'RandomNodeLoader':
                return self.__random_node_loader()
            case 'GraphSAINTRandomWalkSampler':
                return self.__graph_saint_random_walk()
            case _:
                raise DatasetException(f'Graph data loader {self.attributes_map["id"]} is not supported')

    def __str__(self) -> AnyStr:
        return f'\nAttributes: {str(self.attributes_map)}\nData: {self.data}'

    """ ------------------------ Private Helper Methods ------------------------ """

    def __random_node_loader(self) -> DataLoader:
        num_parts = self.attributes_map['num_parts']
        return RandomNodeLoader(self.data, num_parts=num_parts)

    def __neighbors_loader(self) -> DataLoader:
        num_neighbors = self.attributes_map['num_neighbors']
        batch_size = self.attributes_map['batch_size']
        replace = self.attributes_map['replace']
        return NeighborLoader(self.data,
                              num_neighbors=num_neighbors,
                              batch_size=batch_size,
                              replace=replace)

    def __graph_saint_random_walk(self) -> DataLoader:
        walk_length = self.attributes_map['walk_length']
        batch_size = self.attributes_map['batch_size']
        num_steps = self.attributes_map['num_steps']
        return GraphSAINTRandomWalkSampler(data=self.data,
                                           batch_size=batch_size,
                                           walk_length=walk_length,
                                           num_steps=num_steps)
      
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
                case 'GraphSAINTRandomWalkSampler':
                    is_valid = ('walk_length' in attributes_map and
                                'batch_size' in attributes_map and
                                'num_steps' in attributes_map)
        if not is_valid:
            raise DatasetException(GraphDataLoader.__attrs_map_def)

    __attrs_map_def: AnyStr =('Attributes map:\nNeighborLoader: num_neighbors, batch_size, replace'
                              '\nRandomNodeLoader: num_parts\nGraphSAINTRandomWalkSampler: '
                              'walk_length, batch_size, num_steps')






