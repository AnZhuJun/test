# This is a sample Python script.
import torch
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def And(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7

    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(And(0,0))
print(And(1,0))
print(And(0,1))
print(And(1,1))

z = np.arange(-5.0,5.0,0.1)
print(z)
