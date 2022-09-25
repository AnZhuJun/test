import torch
import numpy as np

# a = torch.tensor(1)
#
# print(torch.is_tensor(a))
# print(a)
#
# print(torch.is_storage(a))
# # torch.set_default_tensor_type(t='t')
# # print(a)
#
# a = torch.randn(1, 2, 3, 4, 5)
# print(torch.numel(a))
# a = torch.zeros(4, 4)
# print(torch.numel(a))
#
# print(torch.eye(3))
#
# a = np.array([1, 2, 3])
# t = torch.from_numpy(a)
# print(t)
#
# print(torch.linspace(1, 10, 5))


# print(torch.logspace(1, 2, 5))
# print(torch.ones([3,3]))
# print(torch.rand([2,2]))
# print(torch.randn(3,3))
# print(torch.randperm(5))
# print(torch.arange(1,4))
# print(torch.range(1,4))


x = torch.randn(2,3)
print(torch.cat((x,x,x),0))
print(torch.cat((x,x,x),1))
