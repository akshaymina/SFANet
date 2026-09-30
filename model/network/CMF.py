import torch
import torch.nn as nn


class CMA_Block(nn.Module):
    """Cross-modality attention used by the supplied SFANet code.

    RGB/spatial features produce Q; frequency features produce K and V.
    """

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, 1)
        self.conv2 = nn.Conv2d(in_channel, hidden_channel, 1)
        self.conv3 = nn.Conv2d(in_channel, hidden_channel, 1)
        self.scale = hidden_channel ** -0.5
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()
        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.flatten(2).transpose(1, 2)  # B, HW, Cq
        k = k.flatten(2)                  # B, Ck, HW
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)

        v = v.flatten(2).transpose(1, 2)   # B, HW, Cv
        z = torch.matmul(attn, v)
        z = z.view(z.size(0), h, w, -1).permute(0, 3, 1, 2).contiguous()
        return rgb + self.conv4(z)
