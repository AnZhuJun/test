import torch
from torch import nn
import torch.nn.functional as F


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


# layer = CenteredLayer()
# # print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
#
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# Y = net(torch.rand(4, 8))
# print(Y.mean())


class MyLiner(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        liner = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(liner)


liner = MyLiner(5, 3)
# print(liner.weight)

print(liner(torch.rand(2, 5)))

net = nn.Sequential(nn.Linear(64, 8), nn.Linear(8, 1))
print(net(torch.rand(2, 64)))

