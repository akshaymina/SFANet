from torch import nn
import torch
from torch.nn import Module, Parameter, Softmax


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PAM_Module(Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class DA_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(DA_Fusion, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(in_channels, in_channels, 1),
                                   nn.ReLU())

    def forward(self, x_high, x_low):
        # Processing high-frequency feature map
        feat1 = self.conv5a(x_high)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output1 = self.conv6(sa_conv)

        # Processing low-frequency feature map
        feat2 = self.conv5c(x_low)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output2 = self.conv7(sc_conv)

        # Fuse the outputs from both branches
        feat_sum = sa_output1 + sc_output2

        # Final convolution
        sasc_output = self.conv8(feat_sum)

        return sasc_output

if __name__ == '__main__':
    input_high = torch.randn(1, 256, 19, 19)  # High-frequency input: B C H W
    input_low = torch.randn(1, 256, 19, 19)   # Low-frequency input: B C H W
    print(f"High frequency input size: {input_high.size()}")
    print(f"Low frequency input size: {input_low.size()}")
    block = DA_Fusion(256)
    output = block(input_high, input_low)
    print(f"Output size: {output.size()}")
