import torch
from d2l import torch as d2l
import matplotlib
import pandas as pd

# L1范数 向量元素绝对值之和
# L2范数 向量元素平方和开根号
# Frobenius范数 矩阵元素平方和开根号

#梯度 矩阵A  n维向量x=[x1,x2,...,xn]T
#A：m*n   Ax 相对于x的梯度  A（T）
#A：n*m   x（T）A 相对于x的梯度 A
#A：n*n   x（T)Ax 相对于x的梯度 （A + A（T)）x
#||x||**2 = x（T)x 相对于x的梯度  2x



# x = torch.arange(12)
# print(x)
#
# print(x.shape)
#
# print(x.numel())
#
# X = x.reshape(3,4)
# print(X)
#
# print("tensor x is :",x)
#
# Y = x.reshape(-1,4)
# print(Y)
#
# Z = x.reshape(3,-1)
# print("tensor z is :\n",Z)
#
# M = torch.zeros(3,3,4)
# print("tensor M is :\n",M)
#
# N = torch.ones(2,2,2)
# print("tensor N is :\n",N)
#
# L = torch.randn(3,4)
# print("tensor L is :\n",L)
#
# Q = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print("tensor Q is :\n",Q)
#
# W = torch.tensor([1.0,2,4,8])
# E = torch.tensor([2,2,2,2])
# print("+-*/")
# print(W+E)
# print(W-E)
# print(W*E)
# print(W/E)
#
# x1 = torch.arange(12,dtype=torch.float32).reshape(3,4)
# y1 = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
# print("tensor x1 is :\n",x1)
# print("tensor y1 is :\n",y1)
# x2 = torch.cat((x1,y1))
# x3 = torch.cat((x1,y1),dim=0)
# x4 = torch.cat((x1,y1),dim=1)
# print("tensor x2 is :\n",x2)
# print("tensor x3 is :\n",x3)
# print("tensor x4 is :\n",x4)
# print(x1 == y1)
#
# a1 = torch.arange(3).reshape(3,1)
# print(a1)
# b1 = torch.arange(2).reshape(1,2)
# print(b1)
# print("a1 + b1 is :\n",a1+b1)
#
# print(X)
# print(X[-1])
# print(X[1:3])
# X[1,2]=9
# print(X)
# X[0:2]=12
# print(X)
#
# A = X.numpy()
# B = torch.tensor(A)
# print(type(A))
# print(type(B))
#
# a = torch.tensor(3.5)
# print(a)
# print(a.item())
# print(float(a))
# print(int(a))
#
# x = torch.tensor(3.0)
# y = torch.tensor(2.0)
# print(x+y)
# print(x*y)
# print(x/y)
# print(x**y)
# x = torch.arange(4)
# print(x)
# print(x[3])
# print(len(x))
# print(x.shape)
#
#
# A = torch.arange(20).reshape(5,4)
# print(A)
# print(A.T)
#
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# B = A.clone()
# print(A)
# print(A+B)
# print(A*B)
#
# a = 2
# X = torch.arange(24).reshape(2,3,4)
# print(a + X)
# print((a*X).shape)




#求和
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# print(A.shape)
# print(A.sum())
# print(A)
# A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0)
# print(A_sum_axis0.shape)
# A_sum_axis1 = A.sum(axis=1)
# print(A_sum_axis1)
# print(A_sum_axis1.shape)
# print(A.sum(axis=[0,1]))
# print(A.mean())
# print(A.cumsum(axis=0))




#点积
# x = torch.arange(4,dtype=torch.float32)
# y = torch.ones(4,dtype=torch.float32)
# print(x)
# print(y)
# print(torch.dot(x,y))


# 向量积  MV
# a = torch.arange(20,dtype=torch.float32).reshape(5,4)
# b = torch.arange(4,dtype=torch.float32)
# print(a)
# print(b)
# print(a.shape)
# print(b.shape)
# print(torch.mv(a,b))



#矩阵乘法 MM
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# B = torch.ones(4,3)
# print("matrix A is : \n" ,A)
# print("matrix B is : \n" ,B)
# print(torch.mm(A,B))


#自动微分  y = 2 * xT * x;  2*x**2?
# x = torch.arange(4.0)
# x.requires_grad_(True)
# print(x)
# print(x.grad)
# y = 2 * torch.dot(x,x)
# print(y)
# y.backward()
# print(x.grad)

# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(y)
# print(x.grad)

# x.grad.zero_()
# y = x * x
# y.sum().backward()
# print(x.grad)




#梯度 矩阵A  n维向量x=[x1,x2,...,xn]T
#A：m*n   Ax 相对于x的梯度  A（T）
#A：n*m   x（T）A 相对于x的梯度 A
#A：n*n   x（T)Ax 相对于x的梯度 （A + A（T)）x
#||x||**2 = x（T)x 相对于x的梯度  2x

#分离计算
# x = torch.arange(4.0)
# x.requires_grad = True
# y = x * x
# u = y.detach()
# z = u * x
# z.sum().backward()
# print(x.grad)
# print(u)
#
# x.grad.zero_()
# y.sum().backward()
# print(x.grad)




#python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.tensor([-2.0],requires_grad=True,dtype=float)
d = f(a)
d.backward()
print(a.grad)
