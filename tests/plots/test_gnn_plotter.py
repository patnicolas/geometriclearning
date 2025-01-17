import unittest
from plots.gnn_plotter import GNNPlotter
import networkx as nx


class GNNPlotterTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_sample(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build(_data, sample_node_indices=(12, 21))
        gnn_plotter.sample()
        print(gnn_plotter.graph)

    @unittest.skip('Ignore')
    def test_draw(self):
        from typing import Dict, Any
        from networkx import Graph
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build(_data, sample_node_indices =(12, 21))

        def layout_func(graph: Graph) -> Dict[Any, Any]:
            return nx.spring_layout(graph, k=1)

        gnn_plotter.draw(layout_func=layout_func, node_color='red', node_size=40, title='Flickr spring layout')

        def layout_func(graph: Graph) -> Dict[Any, Any]:
            return nx.random_layout(graph, center=None, dim=2)
        gnn_plotter.draw(layout_func=layout_func, node_color='red', node_size=40, title='Flickr random layout')

    def test_draw_all(self):
        import os
        from torch_geometric.datasets.flickr import Flickr
        from torch.utils.data import Dataset

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Flickr')
        _dataset: Dataset = Flickr(path)
        _data = _dataset[0]

        gnn_plotter = GNNPlotter.build(_data, sample_node_indices=(12, 18))
        gnn_plotter.draw_all(node_size=40, title='Flickr sampled data')
