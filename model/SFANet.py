import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SeparableConv2d, Block
from .xception import xception
from .network.WTFD import WTFD
from .network.CMF import CMA_Block
from .network.Dualfusion import DA_Fusion


class SFANet(nn.Module):
    """Clean, runnable reconstruction of the supplied SFANet core.

    The active forward path follows the provided repository file. Unused
    leftovers from earlier RECCE experiments (GraphReasoning, GuidedAttention,
    FRFN, DA_Block and contrastive-loss storage) are intentionally removed
    because they are not part of the active paper architecture/loss.
    """

    def __init__(self, num_classes=2, drop_rate=0.2, pretrained=False, xception_weights=None):
        super().__init__()
        self.name = "xception"
        self.loss_inputs = dict(recons=[])

        self.encoder = xception(pretrained=pretrained, weights_path=xception_weights)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(2048, num_classes)

        self.wtfd_cat = WTFD(in_ch=256, out_ch=256)
        self.dual_attention = DA_Fusion(256)

        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        # Multi-level feature projection used before WTFD.
        self.encoder.conv_c2 = nn.Conv2d(64, 64, 1)
        self.encoder.conv_b1 = nn.Conv2d(128, 64, 1)
        self.encoder.conv_b2 = nn.Conv2d(256, 128, 1)

        self.cmf = CMA_Block(256, 32, 256)
        self.reconstruction = None

    @staticmethod
    def add_white_noise(tensor, mean=0.0, std=1e-6):
        rand = torch.rand((tensor.shape[0], 1, 1, 1), device=tensor.device)
        rand = (rand > 0.5).float()
        noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        return torch.clamp(tensor + noise * rand, -1.0, 1.0)

    def forward(self, x):
        self.loss_inputs = dict(recons=[])
        noise_x = self.add_white_noise(x) if self.training else x

        out = self.encoder.act1(self.encoder.bn1(self.encoder.conv1(noise_x)))
        out_c1 = out
        del out_c1

        out = self.encoder.act2(self.encoder.bn2(self.encoder.conv2(out)))
        out_c2 = out

        out_b1 = self.encoder.block1(out)
        out_b2 = self.encoder.block2(out_b1)

        out_c2_interp = F.interpolate(out_c2, size=(37, 37), mode="bilinear", align_corners=False)
        out_b1_interp = F.interpolate(out_b1, size=(37, 37), mode="bilinear", align_corners=False)
        out_b2_interp = F.interpolate(out_b2, size=(37, 37), mode="bilinear", align_corners=False)

        out_c2_conv = self.encoder.conv_c2(out_c2_interp)
        out_b1_conv = self.encoder.conv_b1(out_b1_interp)
        out_b2_conv = self.encoder.conv_b2(out_b2_interp)

        out_combined = torch.cat([out_c2_conv, out_b1_conv, out_b2_conv], dim=1)
        combined_L, combined_H = self.wtfd_cat(out_combined)
        da_fusion = self.dual_attention(combined_L, combined_H)

        cmf = self.cmf(out_b2, da_fusion)
        out_b3 = self.encoder.block3(cmf)
        embedding_b4 = self.encoder.block4(out_b3)

        # Reconstruction branch.
        out = self.dropout(embedding_b4)
        out = self.decoder1(out)
        out_d2 = self.decoder2(out)
        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)
        out = self.decoder5(out_d4)
        pred = self.decoder6(out)
        recons_x = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=True)
        self.reconstruction = recons_x
        self.loss_inputs["recons"].append(recons_x)

        # Classification branch: continue through the Xception backbone.
        embedding = self.encoder.block5(embedding_b4)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)
        embedding = self.encoder.block8(embedding)
        embedding = self.encoder.block9(embedding)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)
        embedding = self.encoder.act3(self.encoder.bn3(self.encoder.conv3(embedding)))
        embedding = self.encoder.act4(self.encoder.bn4(self.encoder.conv4(embedding)))

        embedding = self.global_pool(embedding).flatten(1)
        embedding = self.dropout(embedding)
        return self.fc(embedding)
