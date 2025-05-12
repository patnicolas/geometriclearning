import unittest
from plots.gnn_plotter import GNNPlotter
import networkx as nx
import logging

class GNNPlotterTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_sample(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build(_data, sampled_node_index_range=(12, 21))
        gnn_plotter.sample()
        logging.info(gnn_plotter.graph)

    @unittest.skip('Ignore')
    def test_draw(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build_directed(_data, sampled_node_index_range =(12, 21))
        num_sampled_nodes = gnn_plotter.draw(layout_func=lambda graph: nx.spring_layout(graph, k=1),
                                             node_color='blue',
                                             node_size=40,
                                             title='Flickr spring layout')
        logging.info(num_sampled_nodes)


    def test_draw_all_undirected(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build(_data, sampled_node_index_range=(12, 18))
        num_sampled_nodes = gnn_plotter.draw_all(node_size=100, title='Flickr directed')
        logging.info(num_sampled_nodes)

    @unittest.skip('Ignore')
    def test_draw_all_directed(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build_directed(_data, sampled_node_index_range=(12, 18))
        gnn_plotter.draw_all(node_size=100, title='Flickr directed')
