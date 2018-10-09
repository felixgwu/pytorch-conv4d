import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True, drop_connect=0.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) * 4 if isinstance(kernel_size, int) else kernel_size 
        self.padding = (padding,) * 4 if isinstance(padding, int) else padding 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, *self.kernel_size))
        self.drop_connect = drop_connect
        if bias:
            self.bias = nn.Parameter(torch.rand(out_channels)) 
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        k1, k2, k3, k4 = self.kernel_size
        p1, p2, p3, p4 = self.padding
        input_pad = F.pad(input, (p4, p4, p3, p3, p2, p2, p1, p1))
        B, C_in, W2, H2, U2, V2 = input_pad.size()
        assert C_in == self.in_channels
        C_out = self.out_channels

        input_unfold = input_pad.as_strided(
            [B, C_in, k1, k2, k3, k4, W2-k1+1, H2-k2+1, U2-k3+1, V2-k4+1],
            [C_in*W2*H2*U2*V2, W2*H2*U2*V2, H2*U2*V2, U2*V2, V2, 1, H2*U2*V2, U2*V2, V2, 1]
        )
        weight = self.weight
        if self.drop_connect > 0:
            weight = F.dropout(weight, p=self.drop_connect, training=self.training)
        output = torch.einsum('oicdef,bicdefwhuv->bowhuv', (weight, input_unfold))
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{}, {}, kernel_size={}, padding={}, bias={}'.format(
            self.in_channels, self.out_channels,
            self.kernel_size, self.padding, self.bias is not None,
        )
        if self.drop_connect > 0:
            s += ', drop_connect={}'.format(self.drop_connect)
        return s