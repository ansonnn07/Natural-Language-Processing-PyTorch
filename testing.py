import numpy as np
import torch

x = torch.arange(0, 4)
print(x)

y = x.expand((3, 4))
print(y)
print(y.shape)

z = x.repeat((3, 1))
print(z)
print(z.shape)
