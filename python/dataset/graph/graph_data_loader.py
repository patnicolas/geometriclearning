__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from sympy import andre
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.loader import (NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler,
                                    GraphSAINTNodeSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler,
                                    ClusterData, ClusterLoader)
from typing import Dict, AnyStr, Any, Self
from networkx import Graph
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
                 dataset_name: AnyStr) -> None:
        """
        Constructor for the Generic Graph Data Loader
        @param loader_attributes: Map for attributes for a given Data loader
        @type loader_attributes: Dict[AnyStr, Any]
        @param dataset_name: Name of the data set
        @type dataset_name: AnyStr
        """
        from dataset.graph.pyg_datasets import PyGDatasets
        # Validate the attributes against the type of loader-sampler
        GraphDataLoader.__validate(loader_attributes)
        # Load directly from the dataset
        pyg_datasets = PyGDatasets(dataset_name)
        dataset = pyg_datasets()
        self.data = dataset[0]
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

    def to_networkx(self, first_node_index: int, last_node_index: int) -> Graph:
        """
        Generate a graphic representation of the graph defined in this loader
        @param first_node_index: Index of the first node of the graph to be drawn
        @type first_node_index: int
        @param last_node_index: Index of the last node of the graph to be drawn
        @type last_node_index: int
        @return: Instance of Networkx graph
        @rtype: networkx.Graph
        """
        import numpy as np
        import networkx as nx

        # Create NetworkX graph from edge index
        edge_index = self.data.edge_index.numpy()
        transposed = edge_index.T
        # Sample the edges of the graph
        condition = ((transposed[:, 0] >= first_node_index) & (transposed[:, 0] <= last_node_index))
        samples = transposed[np.where(condition)]
        # Create a NetworkX graph
        graph = nx.Graph()
        graph.add_edges_from(samples)
        return graph

    def draw_sample(self,
                    first_node_index: int,
                    last_node_index: int,
                    node_color: AnyStr,
                    node_size: int,
                    label: AnyStr = None) -> int:
        """
        Draw a sample or subgraph of the graph associated with this loader.
        The sample of nodes to be drawn have the range [first_node_index, first_node_index+1, ..., last_node_index]
        @param first_node_index: Index of the first node of the graph to be drawn
        @type first_node_index: int
        @param last_node_index: Index of the last node of the graph to be drawn
        @type last_node_index: int
        @param node_color: Color of the node to be dranw
        @type node_color: AnyStr
        @param node_size: Size of the nodes to be drawn
        @type node_size: int
        @param label: Title or description for the plot
        @type label: AnyStr
        @return Number of nodes in the graph
        @rtype int
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        assert 0 < first_node_index < last_node_index, \
            f'Cannot draw sample with node index in range [{first_node_index}, {last_node_index}]'
        assert 20 < node_size < 512, f'Cannot draw sample with a node size {node_size}'

        graph = self.to_networkx(first_node_index, last_node_index)
        # Plot the graph using matplotlib
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(graph, k=1)  # Spring layout for positioning nodes
        # Draw nodes and edges
        nx.draw_networkx(graph, pos, node_size=node_size, node_color=node_color, with_labels=False)
        nx.draw_networkx_edges(graph, pos, arrowsize=40, alpha=0.5, edge_color="grey")

        # Configure plot
        plt.title(label=label, loc='center', fontdict={'size': 16})
        plt.axis("off")
        plt.show()
        return len(graph)

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
        train_loader = NeighborLoader(self.data,
                                      num_neighbors=num_neighbors,
                                      batch_size=batch_size,
                                      replace=replace,
                                      drop_last=False,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      input_nodes=self.data.train_mask)
        val_loader = NeighborLoader(self.data,
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
                                'sample_coverage' in attributes_map)

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






