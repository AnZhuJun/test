import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这⾥， module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。 module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.liner = nn.Linear(20, 20)

    def forward(self, X):
        X = self.liner(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.liner(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# X = torch.rand(2, 20)
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(X))

# net = MLP()
# print(net(X))
