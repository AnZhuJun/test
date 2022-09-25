import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
x = torch.rand(size=(2,4))
# print(net(x))
# print(net)
# print(net[0].state_dict())
# print(net[1].state_dict())
# print(net[2].state_dict())

# print(*[(name,param.shape) for name,param in net[0].named_parameters()])
# print(*[(name,param.shape) for name,param in net.named_parameters()])
# print(net.state_dict()['2.weight'].data)
# print(net.state_dict()['2.bias'].data)



# def block1():
#     return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
#
# def block2():
#     net = nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block{i}',block1())
#     return net
#
# rgnet = nn.Sequential(block2(),nn.Linear(4,1))
# print(rgnet(x))
#
# print(rgnet)



def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)
# net.apply(init_normal)
# print(net[0].weight.data[0],net[0].bias.data[0])

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
# net.apply(init_constant)
# print(net[0].weight.data[0],net[0].bias.data[0])

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

# net[0].apply(xavier)
# net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight)

def my_init(m):
    if type(m) == nn.Linear:
        print("init ",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)
print(net[0].weight)


