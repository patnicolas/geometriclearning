
from typing import AnyStr, Any, Dict
import logging
import python
import unittest
import torch
import torch.nn as nn
from topology.topo_bench_wrapper import TopoBenchWrapper


class TestGraphNetwork(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.linear_0 = torch.nn.Linear(in_channels, hidden_size)
        self.linear_1 = torch.nn.Linear(in_channels, hidden_size)
        self.out_channels = out_channels

    def get_dim_hidden(self) -> int:
        return self.linear_0.out_features

    def forward(self, batch) -> Dict[AnyStr, Any]:
        x_0 = batch.x
        x_1 = torch.sparse.mm(batch.incidence_hyperedges, x_0)

        x_0 = self.linear_0(x_0)
        x_0 = torch.relu(x_0)
        x_1 = self.linear_1(x_1)
        x_1 = torch.relu(x_1)
        return {"labels": batch.y, "batch_0": batch.batch_0, "x_0": x_0, "hyperedge": x_1}


class TopoBenchWrapperTest(unittest.TestCase):

    def test_init(self):
        dim_hidden = 24
        k = 2
        model = TestGraphNetwork(hidden_size=dim_hidden, in_channels=7, out_channels=2)
        topo_bench_wrapper = TopoBenchWrapper.build(graph_network=model, k=k)
        logging.info(topo_bench_wrapper)

    @unittest.skip('Ignored')
    def test_train(self):
        dim_hidden = 24
        k = 2
        model = TestGraphNetwork(hidden_size=dim_hidden, in_channels=7, out_channels=2)
        topo_bench_wrapper = TopoBenchWrapper.build(graph_network=model, k=k)
        topo_bench_wrapper.train(max_epochs=24, float_precision=32)