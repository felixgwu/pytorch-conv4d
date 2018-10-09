import torch
import time
import torch.nn.functional as F

k = 3
B, C_in, W, H, U, V = 32, 4, 5, 6, 7, 8
C_out = 2

weight = torch.rand(C_out, C_in, k, k, k, k)
x = torch.rand(B, C_in, W, H, U, V)

start = time.perf_counter()
x_pad = F.pad(x, [k//2]*8)
W2, H2, U2, V2 = W + k - 1, H + k - 1, U + k - 1, V + k - 1
x_unfold = x_pad.as_strided([B, C_in, k, k, k, k, W, H, U, V], [C_in*W2*H2*U2*V2, W2*H2*U2*V2, H2*U2*V2, U2*V2, V2, 1, H2*U2*V2, U2*V2, V2, 1])

end = time.perf_counter()
print('unfold time:', end - start)

start = time.perf_counter()

print(weight.size(), x_unfold.size())
o = torch.einsum('oicdef,bicdefwhuv->bowhuv', (weight, x_unfold))
# o = (weight.view(1, C_out, C_in, k ,k, k, k, 1, 1, 1, 1) * x_unfold.unsqueeze(1)).view(B, C_out, C_in * k**4, W, H, U, V).sum(2)
print(o.size())

end = time.perf_counter()
print('conv time:', end - start)

# start = time.perf_counter()
# x_pad = F.pad(x, [k//2]*8)
# W2, H2, U2, V2 = W + k - 1, H + k - 1, U + k - 1, V + k - 1
# x_unfold = x_pad.as_strided([B, C_in, W, H, U, V, k, k, k, k], [C_in*W2*H2*U2*V2, W2*H2*U2*V2, H2*U2*V2, U2*V2, V2, 1, H2*U2*V2, U2*V2, V2, 1])

# end = time.perf_counter()
# print('unfold time:', end - start)

# start = time.perf_counter()

# print(weight.size(), x_unfold.size())
# # o = torch.einsum('oicdef,biwhuvcdef->bowhuv', (weight, x_unfold))
# o = (weight.view(1, C_out, C_in, 1, 1, 1, 1, k ,k, k, k) * x_unfold.unsqueeze(1)).view(B, C_out, C_in, W, H, U, V, k**4).sum(-1).sum(2)
# print(o.size())

# end = time.perf_counter()
# print('conv time:', end - start)
