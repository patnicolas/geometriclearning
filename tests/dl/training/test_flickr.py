import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # regular layer
        # self.activation = nn.ReLU() # regular layer
        self.linear2 = nn.Linear(5, 2)  # regular layer
        # self.custom_param = nn.Parameter(torch.randn(5))  # custom trainable param

    def forward(self, x):
        x = self.linear(x)
        return x + self.custom_param  # use custom param in forward pass


if __name__ == '__main__':
    model = MyModel()
    params = model.parameters()
    lst = list(params)
    print(lst)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(optimizer)



