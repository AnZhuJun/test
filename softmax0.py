import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
W = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)


# 1. 对每个项求幂（使⽤exp）；
# 2. 对每⼀⾏求和（⼩批量中每个样本是⼀⾏），得到每个样本的规范化常数；
# 3. 将每⼀⾏除以其规范化常数，确保结果的和为1。

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim = True)
    #1为对行求和，0为对列求和
    return X_exp / partition

X = torch.normal(0,1,(2,5))
x_prob = softmax(X)

#模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])) , W) + b)

#损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print(y_hat[[0,1],y])

def accuracy(y_hat,y):
    #计算预测正确的数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

