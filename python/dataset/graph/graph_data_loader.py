__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."


from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import (NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler,
                                    GraphSAINTNodeSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler,
                                    ClusterData, ClusterLoader)
from typing import Dict, AnyStr, Any, Optional
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
    # Static definition of the sampling method dictionary
    loader_sampler_dict = {
        'NeighborLoader': lambda loader: loader.__neighbors_loader(),
        'RandomNodeLoader': lambda loader: loader.__random_node_loader(),
        'GraphSAINTNodeSampler': lambda loader: loader.__graph_saint_node_sampler(),
        'GraphSAINTEdgeSampler': lambda loader: loader.__graph_saint_edge_sampler(),
        'ShaDowKHopSampler': lambda loader: loader.__shadow_khop_sampler(),
        'GraphSAINTRandomWalkSampler': lambda loader: loader.__graph_saint_random_walk(),
        'ClusterLoader': lambda loader: loader.__cluster_loader(),
    }

    def __init__(self,
                 loader_attributes: Dict[AnyStr, Any],
                 dataset_name: AnyStr,
                 num_subgraph_nodes: Optional[int] = -1,
                 start_index: Optional[int] = -1) -> None:
        """
        Constructor for the Generic Graph Data Loader
        @param loader_attributes: Map for attributes for a given Data loader
        @type loader_attributes: Dict[AnyStr, Any]
        @param dataset_name: Name of the data set
        @type dataset_name: AnyStr
        @param num_subgraph_nodes: Num of indices of nodes in the range [random_index, random_index+num_random_indices]
                                   The entire graph is loaded if value is -1
        @type dataset_name: int
        """
        from dataset.graph.pyg_datasets import PyGDatasets
        # Validate the attributes against the type of loader-sampler
        GraphDataLoader.__validate(loader_attributes)
        # Load directly from the dataset
        pyg_datasets = PyGDatasets(dataset_name)
        dataset = pyg_datasets()
        # Load a subgraph is specified
        self.data: Data = GraphDataLoader.__extract_subgraph(dataset[0], num_subgraph_nodes, start_index) \
            if num_subgraph_nodes > 0 else dataset[0]
        self.attributes_map = loader_attributes

    def __call__(self) -> (DataLoader, DataLoader):
        """
        Generate the data loader for both training and evaluation set
        @return: A pair of training loader and evaluation loader
        @rtype: Tuple[DataLoader, DataLoader]
        """
        sampler_name = self.attributes_map['id']
        loader_sampler = GraphDataLoader.loader_sampler_dict[sampler_name]
        return loader_sampler(self)

    def __str__(self) -> AnyStr:
        return f'\nAttributes: {str(self.attributes_map)}\nData: {self.data}'

    def __len__(self) -> int:
        return len(self.data.x)

    """ ------------------------ Private Helper Methods ------------------------ """

    def __random_node_loader(self) -> (DataLoader, DataLoader):
        # Collect the attributes
        num_parts = self.attributes_map['num_parts']
        num_workers = self.attributes_map['num_workers'] if 'num_workers' in self.attributes_map \
            else 1

        # Use multiple workers for training, non-default batch size and shuffling
        train_loader = RandomNodeLoader(self.data,
                                        num_parts=num_parts,
                                        shuffle=True,
                                        num_workers=num_workers)

        # We use a single GPU for evaluation with default batch size
        eval_loader = RandomNodeLoader(self.data,
                                       num_parts=num_parts,
                                       shuffle=False)
        return train_loader, eval_loader

    def __neighbors_loader(self) -> (DataLoader, DataLoader):
        num_neighbors = self.attributes_map['num_neighbors']
        batch_size = self.attributes_map['batch_size']
        replace = self.attributes_map['replace']
        num_workers = self.attributes_map['num_workers']
        train_loader = NeighborLoader(data=self.data,
                                      num_neighbors=num_neighbors,
                                      batch_size=batch_size,
                                      replace=replace,
                                      drop_last=False,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      input_nodes=self.data.train_mask)
        val_loader = NeighborLoader(data=self.data,
                                    num_neighbors=num_neighbors,
                                    batch_size=batch_size,
                                    replace=replace,
                                    drop_last=False,
                                    shuffle=False,
                                    input_nodes=self.data.val_mask)
        return train_loader, val_loader

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

        # Dynamic configuration parameter for the loader
        walk_length = self.attributes_map['walk_length']
        batch_size = self.attributes_map['batch_size']
        num_steps = self.attributes_map['num_steps']
        sample_coverage = self.attributes_map['sample_coverage']
        num_workers = self.attributes_map['num_workers']

        # Extraction of the loader for training data
        train_loader = GraphSAINTRandomWalkSampler(data=self.data,
                                                   batch_size=batch_size,
                                                   walk_length=walk_length,
                                                   num_steps=num_steps,
                                                   sample_coverage=sample_coverage,
                                                   num_workers=num_workers,
                                                   shuffle=True)

        # Extraction of the loader for validation data
        val_loader = GraphSAINTRandomWalkSampler(data=self.data,
                                                 batch_size=batch_size,
                                                 walk_length=walk_length,
                                                 num_steps=num_steps,
                                                 sample_coverage=sample_coverage,
                                                 num_workers=num_workers,
                                                 shuffle=False)
        return train_loader, val_loader

    def __shadow_khop_sampler(self) -> (DataLoader, DataLoader):
        depth = self.attributes_map['depth']
        num_neighbors = self.attributes_map['num_neighbors']
        batch_size = self.attributes_map['batch_size']
        num_workers = self.attributes_map['num_workers']
        train_loader = ShaDowKHopSampler(data=self.data,
                                         depth=depth,
                                         num_neighbors=num_neighbors,
                                         node_idx=self.data.train_mask,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=True)
        eval_loader = ShaDowKHopSampler(data=self.data,
                                        depth=depth,
                                        num_neighbors=num_neighbors,
                                        node_idx=self.data.val_mask,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
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
    def __extract_subgraph(data: Data, num_random_indices: int, start_index: int) -> Data:
        assert num_random_indices > 0, f'Number of random indices {num_random_indices} should be >0'
        import torch
        import random
        from torch_geometric.utils import subgraph

        dataset_len = len(data.x)

        # Select the starting index as a random value
        start_index = random.randint(0, dataset_len-num_random_indices-1) if start_index == -1 else start_index
        print(f'Start index: {start_index}')

        # Collect the indices of the selected node for the subgraph
        subset = torch.arange(start_index, start_index+num_random_indices)
        edge_index, edge_attr = subgraph(subset=subset,
                                         edge_index=data.edge_index,
                                         edge_attr=data.edge_attr if 'edge_attr' in data else None,
                                         relabel_nodes=True)
        x_sub = data.x[subset]
        y_sub = data.y[subset] if 'y' in data else None

        # Remap the various mask if necessary
        train_mask = data.train_mask[subset] if 'train_mask' in data else None
        val_mask = data.val_mask[subset] if 'val_mask' in data else None
        test_mask = data.test_mask[subset] if 'test_mask' in data else None

        # Finally, create the new subgraph as a Data object
        return Data(x=x_sub,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y_sub,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask)

    @staticmethod
    def __validate(attributes_map: Dict[AnyStr, Any]) -> None:
        is_valid = False
        if attributes_map is not None and 'id' in attributes_map:
            match attributes_map['id']:
                case 'NeighborLoader':
                    is_valid = ('num_neighbors' in attributes_map and
                                'batch_size' in attributes_map and
                                'num_workers' in attributes_map and
                                'replace' in attributes_map)

                case 'RandomNodeLoader':
                    is_valid = ('num_parts' in attributes_map and
                                'batch_size' in attributes_map and
                                'num_workers' in attributes_map)

                case 'GraphSAINTNodeSampler' | 'GraphSAINTEdgeSampler':
                    is_valid = ('batch_size' in attributes_map and
                                'num_steps' in attributes_map and
                                'sample_coverage' in attributes_map)

                case 'GraphSAINTRandomWalkSampler':
                    is_valid = ('walk_length' in attributes_map and
                                'batch_size' in attributes_map and
                                'num_steps' in attributes_map and
                                'sample_coverage' in attributes_map and
                                'num_workers' in attributes_map)

                case 'ShaDowKHopSampler':
                    is_valid = ('depth' in attributes_map and
                                'num_neighbors' in attributes_map and
                                'batch_size' in attributes_map)

                case 'ClusterLoader':
                    is_valid = ('num_parts' in attributes_map and
                                'recursive' in attributes_map and
                                'batch_size' in attributes_map and
                                'keep_inter_cluster_edges' in attributes_map)
        if not is_valid:
            raise DatasetException(GraphDataLoader.__attrs_map_def)

    __attrs_map_def: AnyStr =('Attributes map:\nNeighborLoader: num_neighbors, batch_size, replace'
                              '\nRandomNodeLoader: num_parts\nGraphSAINTRandomWalkSampler: walk_length, batch_size'
                              'num_steps, sample_coverage\nGraphSAINTNodeSampler: batch_size, num_steps sample_coverage'
                              '\nGraphSAINTEdgeSampler: batch_size num_steps sample_coverage\nShaDowKHopSampler: depth'
                              'num_neighbors node_idx\nClusterLoader: num_parts, recursive batch_size'
                              'keep_inter_cluster_edges')






