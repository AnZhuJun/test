import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

def load_array(data_arrays,batch_size,is_train = True):
    #构造一个pytorch数据迭代器
    #布尔值is_train表⽰是否希望数据迭代器对象在每个迭代周期内打乱数据
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

#1.初始化生成数据
true_w = torch.tensor([2,-3.4])
trub_b = 4.2
features,labels = d2l.synthetic_data(true_w,trub_b,1000)

#2.读取数据
batch_size = 10
data_iter = load_array((features,labels),batch_size)
# print(next(iter(data_iter)))

#3.指定模型，随机初值
net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
# print(net[0].weight.data)
# print(net[0].bias.data)

#4.定义损失函数和梯度下降算法
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

#5.训练
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch{epoch + 1},loss{l:f}')

w = net[0].weight.data
b = net[0].bias.data
print('w的估计误差:',true_w - w.reshape(true_w.shape))
print('w的估计误差:',trub_b - b)


