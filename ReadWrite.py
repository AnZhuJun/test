import torch
from torch import nn
from torch.nn import functional as F

# x = torch.arange(4)
# torch.save(x, 'x-file')
# x2 = torch.load('x-file')
# # print(x2)
#
#
# y = torch.zeros(4)
# torch.save([x, y], 'x-files')
# x2, y2 = torch.load('x-files')
# print(x2, y2)
#
#
# my_dict = {'x': x, 'y': y}
# torch.save(my_dict, 'my_dict')
# my_dict2 = torch.load('my_dict')
# print(my_dict2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)
