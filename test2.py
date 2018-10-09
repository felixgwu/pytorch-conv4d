import torch
from conv4d import Conv4d

k = 3
B, C_in, W, H, U, V = 32, 4, 5, 6, 7, 8
C_out = 2

x = torch.rand(B, C_in, W, H, U, V)
m = Conv4d(C_in, C_out, (7, 5, 3, 1), (3, 2, 1, 0), bias=True)
print(m)

print('input size:', x.size())
o = m(x)

print('output size:', o.size())

