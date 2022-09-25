import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x,'x-files')
x2 = torch.load('x-files')
print(x2)

y = torch.zeros(4)
mydict = {'x':x,'y':y}
torch.save(mydict,'mydict')
z = torch.load('mydict')
print(z)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
torch.save(net.state_dict(),'mlp.params1')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params1'))
print(clone.eval())
