import unittest
from torch.utils.data import Dataset
from dataset import DatasetException
from torch_geometric.data import Data
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import (NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler,
                                    GraphSAINTNodeSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler,
                                    ClusterData, ClusterLoader)


class GraphDataLoaderReview(unittest.TestCase):

    def test_neighbors_node_loader(self):
        import os
        from torch_geometric.datasets.flickr import Flickr

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        # Load the dataset as a collection of graph data
        _dataset: Dataset = Flickr(path)
        # Retrieve the first graph data
        data = _dataset[0]
        try:
            num_neighbors = [3, 2]
            batch_size = 16
            replace = True
            drop_last = True

            train_loader, val_loader = GraphDataLoaderReview.neighbors_node_loader(
                data, num_neighbors, batch_size, replace, drop_last
            )
            first_3_batches = [f'{idx}: {str(batch)}'
                               for idx, batch in enumerate(train_loader) if idx < 3]

            print(f'\n{data}\nFirst 3 batches:')
            print("\n".join(first_3_batches))
        except Exception as e:
            print(f'Type of exception {type(e)}')
            print(f'Exception {e}')


    @staticmethod
    def neighbors_node_loader(data: Data,
                              num_neighbors: int,
                              batch_size: int,
                              replace: bool,
                              drop_last_batch: bool = False) -> (DataLoader, DataLoader):
        # Generate the loader for training data (shuffle True)
        train_loader = NeighborLoader(data,
                                      num_neighbors=num_neighbors,
                                      batch_size=batch_size,
                                      replace=replace,
                                      drop_last=drop_last_batch,
                                      shuffle=True,
                                      input_nodes=data.train_mask)

        # Generate the loader for validation data (shuffle False)
        val_loader = NeighborLoader(data,
                                    num_neighbors=num_neighbors,
                                    batch_size=batch_size,
                                    replace=replace,
                                    drop_last=drop_last_batch,
                                    shuffle=False,
                                    input_nodes=data.val_mask)
        return train_loader, val_loader

