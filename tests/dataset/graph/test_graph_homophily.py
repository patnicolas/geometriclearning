import unittest

from torch_geometric.data.remote_backend_utils import num_nodes

from dataset.graph.graph_homophily import GraphHomophily, GraphHomophilyType
from torch_geometric.data import Data
import torch


class GraphHomophilyTest(unittest.TestCase):

    def test_init_1(self):
        homophily_flickr = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.Node)
        print(homophily_flickr)

    def test_init_2(self):
        homophily_cora = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.Node)
        print(homophily_cora)

    def test_homegrown_edge_homophily(self):
        labels = torch.Tensor([0, 0, 1, 1, 0])
        data = Data(y=labels,
                    edge_index=torch.tensor([
                                        [0, 1, 2, 3],
                                        [1, 2, 3, 4]]),
                    num_nodes=len(labels)
                    )
        homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Edge)
        edge_homophily = homophily.compute()
        print(f'Edge homophily: {edge_homophily}')

    def test_homegrown_node_homophily(self):
        labels = torch.Tensor([0, 0, 1, 1, 0])
        data = Data(y=labels,
                    edge_index=torch.tensor([
                                        [0, 1, 2, 3],
                                        [1, 2, 3, 4]]),
                    num_nodes=len(labels)
                    )
        homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Node)
        node_homophily = homophily.compute()
        print(f'Node homophily: {node_homophily}')

    def test_edge_homophily(self):
        labels = torch.Tensor([0, 0, 1, 1, 0])
        data = Data(y=labels,
                    edge_index=torch.tensor([
                                        [0, 1, 2, 3],
                                        [1, 2, 3, 4]]),
                    num_nodes=len(labels)
                    )
        homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Edge)
        edge_homophily = homophily()
        print(f'Edge homophily: {edge_homophily}')

    def test_node_homophily(self):
        labels = torch.Tensor([0, 0, 1, 1, 0])
        data = Data(y=labels,
                    edge_index=torch.tensor([
                                        [0, 1, 2, 3],
                                        [1, 2, 3, 4]]),
                    num_nodes=len(labels)
                    )
        homophily = GraphHomophily(data=data, homophily_type=GraphHomophilyType.Node)
        node_homophily = homophily()
        print(f'Node homophily: {node_homophily}')


    def test_node_homophily_datasets(self):
        homophily = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.Node)
        node_homophily = homophily()
        print(f'Flickr node homophily: {node_homophily:.3f}')

        homophily = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.Node)
        node_homophily = homophily()
        print(f'Cora node homophily: {node_homophily:.3f}')

    def test_edge_homophily_datasets(self):
        homophily = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.Edge)
        edge_homophily = homophily()
        print(f'Flickr edge homophily: {edge_homophily:.3f}')

        homophily = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.Edge)
        edge_homophily = homophily()
        print(f'Cora edge homophily: {edge_homophily:.3f}')

    def test_class_insensitive_edge_homophily_datasets(self):
        homophily = GraphHomophily.build(dataset_name='Flickr', homophily_type=GraphHomophilyType.ClassInsensitiveEdge)
        edge_homophily = homophily()
        print(f'Flickr class insensitive edge homophily: {edge_homophily:.3f}')

        homophily = GraphHomophily.build(dataset_name='Cora', homophily_type=GraphHomophilyType.ClassInsensitiveEdge)
        edge_homophily = homophily()
        print(f'Cora class insensitive edge homophily: {edge_homophily:.3f}')

