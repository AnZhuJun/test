import torch
import torch.nn.functional as F
from torch import nn

class CerterdLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return X - X.mean()

# layer = CerterdLayer()
# print(layer(torch.FloatTensor([1,2,3,4,5])))
# net = nn.Sequential(nn.Linear(8,128),CerterdLayer())
# Y = net(torch.rand(4,8))
# print(Y.mean())


class MyLiner(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self,X):
        liner = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(liner)

linear = MyLiner(5,3)
print(linear.weight)
print(linear.bias)
print(linear(torch.rand(2,5)))
net = nn.Sequential(MyLiner(64,8),MyLiner(8,1))
print(net(torch.rand(2,64)))
