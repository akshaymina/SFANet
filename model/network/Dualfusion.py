import torch
from torch import nn
from torch.nn import Module, Parameter, Softmax


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class PAM_Module(Module):
    """Position attention module."""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = DepthWiseConv2d(in_dim, in_dim, 1)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim, 1)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim, 1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query_conv(x).view(b, c, h * w).permute(0, 2, 1)
        k = self.key_conv(x).view(b, c, h * w)
        attention = self.softmax(torch.bmm(q, k))
        v = self.value_conv(x).view(b, c, h * w)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class CAM_Module(Module):
    """Channel attention module."""
    def __init__(self, in_dim):
        super().__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = x.view(b, c, -1)
        k = q.permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = energy.max(-1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        value = x.view(b, c, -1)
        out = torch.bmm(attention, value).view(b, c, h, w)
        return self.gamma * out + x


class DA_Fusion(nn.Module):
    """Dual-attention fusion of high- and low-frequency feature maps."""
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 16
        self.conv5a = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv5c = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv52 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(in_channels, in_channels, 1), nn.ReLU())

    def forward(self, x_high, x_low):
        feat1 = self.conv5a(x_high)
        sa_output1 = self.conv6(self.conv51(self.sa(feat1)))

        feat2 = self.conv5c(x_low)
        sc_output2 = self.conv7(self.conv52(self.sc(feat2)))

        return self.conv8(sa_output1 + sc_output2)
