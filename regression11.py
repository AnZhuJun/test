import torch
import numpy as np
from torch import nn
from d2l import torch as d2l
from torch.utils import data

def load_array(data_array,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

batch_size = 10
data_iter = load_array((features,labels),batch_size)

net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

epochs = 3
for epoch in range(epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch : {epoch + 1},loss : {l : f}')

w = net[0].weight.data
b = net[0].bias.data
print("w误差:",true_w - w.reshape(true_w.shape))
print("w误差:",true_b - b)


