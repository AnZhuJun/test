import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas
def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def identify_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

# def cross_entropy_error(y,t):
#     delta = 1e - 7
#     return -np.sum(np.log(y + delta))

# X = np.array([1.0,0.5])
# W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
# B1 = np.array([0.1,0.2,0.3])
#
# A1 = np.dot(X,W1)+B1
# Z1 = sigmoid(A1)
#
# print(A1)
# print(Z1)
#
# W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
# B2 = np.array([0.1,0.2])
# A2 = np.dot(Z1,W2)+B2
# Z2 = sigmoid(A2)
#
# print("A2: " ,A2)
# print("Z2: " ,Z2)
#
# W3 = np.array([[0.1,0.3],[0.2,0.4]])
# B3 = np.array([0.1,0.2])
# A3 = np.dot(Z2,W3)+B3
# Y = identify_function(A3)
# print(Y)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forword(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2,W3)+b3
    y = identify_function(a3)

    return y

# network = init_network()
# x = np.array([1.0,0.5])
# y = forword(network,x)
# print(y)

a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
