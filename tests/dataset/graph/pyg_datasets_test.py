import unittest
import logging
from dataset.graph.pyg_datasets import PyGDatasets
from torch_geometric.data import Data
from typing import AnyStr
import python


class PyGDatasetsTest(unittest.TestCase):

    def test_cite_seer(self):
        PyGDatasetsTest.__load_show_data('CiteSeer')
        self.assertTrue(True)

    @staticmethod
    def __load_show_data(name: AnyStr) -> None:
        pyg_dataset = PyGDatasets(name)
        _dataset = pyg_dataset()
        data = _dataset[0]
        sub_data = PyGDatasetsTest.__extract_subgraph(data)

        # sub_data.node_mapping = subset_nodes
        logging.info(f'\n{name}:\n{sub_data}')

    @staticmethod
    def __extract_subgraph(data: Data) -> Data:
        import torch
        from torch_geometric.utils import k_hop_subgraph

        dataset_len = len(data.x)
        subset_nodes = torch.randperm(dataset_len)[:10]
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=subset_nodes,
            num_hops=2,
            edge_index=data.edge_index,
            relabel_nodes=True
        )
        return Data(x=data.x[subset_nodes],
                    edge_index=edge_index,
                    y=data.y[subset_nodes],
                    train_mask=data.train_mask[subset_nodes],
                    val_mask=data.val_mask[subset_nodes],
                    node_mapping=subset_nodes)