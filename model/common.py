"""Small Xception-style building blocks used by SFANet.

The original SFANet file imports SeparableConv2d and Block from
``model.common``. These implementations match the Xception-style blocks
used by the authors' code lineage (RECCE / Cadene Xception).
"""

import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    """Xception residual block used both by the backbone and decoder."""

    def __init__(self, in_channels, out_channels, reps, strides=1,
                 start_with_relu=True, grow_first=True, with_bn=True):
        super().__init__()
        self.with_bn = with_bn

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_channels) if with_bn else nn.Identity()
        else:
            self.skip = None
            self.skipbn = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels

            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, padding=1, bias=False))
            if with_bn:
                rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu and rep:
            rep = rep[1:]
        elif rep:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, stride=strides, padding=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is None:
            skip = x
        else:
            skip = self.skipbn(self.skip(x))
        return out + skip
