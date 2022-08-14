import numpy as np


#np.ndim()数组维度
#np.shape 数组大小
#np.dot(A,B) A,B矩阵的乘积
#np.sum(nparray) 求和
#np.exp(a) e的a次方



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0,1.0,2.0])
print(sigmoid(x))