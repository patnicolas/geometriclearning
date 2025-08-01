import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGETrainer:
    def __init__(self, data, model, lr=0.01, weight_decay=5e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        preds = out.argmax(dim=1)
        accs = []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            correct = preds[mask].eq(self.data.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
        return accs  # train_acc, val_acc, test_acc


# Example usage
if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    model = GraphSAGE(in_channels=dataset.num_node_features,
                      hidden_channels=64,
                      out_channels=dataset.num_classes,
                      num_layers=2)

    trainer = GraphSAGETrainer(data, model)

    for epoch in range(1, 201):
        loss = trainer.train()
        train_acc, val_acc, test_acc = trainer.evaluate()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
