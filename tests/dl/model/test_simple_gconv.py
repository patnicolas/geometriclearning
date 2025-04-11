import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, global_mean_pool

class HierarchicalGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)

        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)

        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1st conv + pooling
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        # 2nd conv + pooling
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)

        return self.fc(x)

