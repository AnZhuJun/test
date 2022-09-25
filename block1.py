import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
# print(net(X))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X))


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20), requires_grad=True)
        self.liner = nn.Linear(20, 20)

    def forward(self,X):
        X = self.liner(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.liner(X)
        while X.abs().sum() > 1:
            X = X/2
        return X.sum()

