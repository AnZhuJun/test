import torch
from torch import nn

print(torch.cuda.device_count())

x = torch.tensor([1, 2, 3])
print(x.device)

