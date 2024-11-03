import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from typing import Callable
from tqdm import tqdm


"""
0: Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
1: BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
2: ELU(alpha=1.0)
3: MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
4: Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
5: BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
6: MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
7: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
8: BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
9: MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
10: Flatten(start_dim=1, end_dim=-1)
11: Linear(in_features=46208, out_features=128, bias=False)
12: ReLU()
13: Linear(in_features=128, out_features=10, bias=False)
14: Softmax(dim=1)

"""

class ConvNet(nn.Module):
    num_classes = 10

    def __init__(self, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.15)
        in_fc = 46208
        self.fc1 = nn.Linear(in_features=in_fc, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=ConvNet.num_classes)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.activation(x)
        x = self.dropout(x)
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.activation(x)
        x = self.dropout(x)
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.activation(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        # First FFNN block
        x = self.fc1(x)
        x = self.activation (x)
        # Last layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    @staticmethod
    def training(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()



    @staticmethod
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)



class ConvMNISTVerifyTest(unittest.TestCase):

    def test_validation(self):
        use_cuda = False
        use_mps = True
        torch.manual_seed(42)

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        batch_size = 64
        test_batch_size = 64
        train_kwargs = {'batch_size': batch_size}
        test_kwargs = {'batch_size': test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                               'pin_memory': True,
                               'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        root_path = '../../../../data/MNIST'
        train_data = torch.load(f'{root_path}/processed/training.pt')
        train_features = train_data[0]
        train_labels = train_data[1]
        test_data = torch.load(f'{root_path}/processed/test.pt')
        test_features = test_data[0]
        test_labels = test_data[1]

        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        model = Net().to(device)
        lr = 1.0
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        epochs = 14
        for epoch in range(1, epochs + 1):
            Net.training(model, device, train_loader, optimizer, epoch)
            Net.test(model, device, test_loader)
            scheduler.step()

        save_model = False
        if save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    unittest.main()