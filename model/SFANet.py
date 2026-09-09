import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from timm.models import xception
from model.common import SeparableConv2d, Block
from model.common import GuidedAttention, GraphReasoning
from model.network.WTFD import WTFD
from model.network.CMF import CMA_Block
from model.network.Dualfusion import DA_Fusion


encoder_params = {
    "xception": {
        "features": 2048, 
        "init_op": partial(xception, pretrained=True) 
    }
}

class SFANet(nn.Module):
    def __init__(self, num_classes, drop_rate=0.2):
        super(SFANet, self).__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)
        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        self.reasoning = GraphReasoning(728, 256, 256, 256, 128, 256, [2, 4], drop_rate)

        # self.wtfd_input = WTFD(in_ch=3, out_ch=32)  # 对输入分解
        # self.wtfd_input = WTFD(in_ch=3, out_ch=3)
        # self.wtfd_block3 = WTFD(in_ch=728, out_ch=728)  # 对 block3 输出分解
        # self.wtfd_decoder = WTFD(in_ch=128, out_ch=128)  # 对 decoder 特征分解
        self.wtfd_cat = WTFD(in_ch=256, out_ch=256) 
        

        self.weight_L = nn.Parameter(torch.tensor(0.5))
        self.weight_H = nn.Parameter(torch.tensor(0.5))
        
        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2), 
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.dual_attention1 = DA_Block(128)
        self.dual_attention2 = DA_Block(256)
        self.dual_attention3 = DA_Block(728)

        self.frfn = FRFN(64)

        self.encoder.conv_c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.encoder.conv_b1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.encoder.conv_b2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

        self.cmf = CMA_Block(in_channel=256 , hidden_channel=32, out_channel=256)
        self.DA_fusion = DA_Fusion(256)


    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr
    
    @staticmethod 
    def add_white_noise(tensor, mean=0., std=1e-6): 
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def forward(self, x): # x.shape([32, 3, 299, 299])
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x
        # 输入级小波分解
        # low, high = self.wtfd_input(x)
        # WTFD_x = low + high
        out = self.encoder.conv1(noise_x)# torch.Size([32, 32, 149, 149])
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out_c1 = out
        
        out = self.encoder.conv2(out) # torch.Size([32, 64, 147, 147])
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out_c2 = out
        
        # B, C, H, W = out.shape
        # out_reshaped = out.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # out = self.frfn(out_reshaped)

        # '''中间层xiaobo'''
        # out_b1 = self.encoder.block1(out)  # torch.Size([32, 128, 74, 74])
        # out_b2 = self.encoder.block2(out_b1)  # torch.Size([32, 256, 37, 37])
        # out_b3 = self.encoder.block3(out_b2)  # torch.Size([32, 728, 19, 19])
        # low_b3, high_b3 = self.wtfd_block3(out_b3)  # WTFD 分解 # torch.Size([8, 728, 10, 10])
        # out_hl = self.weight_L * low_b3 + self.weight_H * high_b3
        # out = out_b3 + out_hl
    
        # out_b3 = low_b3 + high_b3

        out_b1 = self.encoder.block1(out) # torch.Size([32, 128, 74, 74])
        out_b2 = self.encoder.block2(out_b1) # torch.Size([32, 256, 37, 37])

        out_c2_interp = F.interpolate(out_c2, size=(37, 37), mode='bilinear', align_corners=False)
        out_b1_interp = F.interpolate(out_b1, size=(37, 37), mode='bilinear', align_corners=False)
        out_b2_interp = F.interpolate(out_b2, size=(37, 37), mode='bilinear', align_corners=False)

        out_c2_conv = self.encoder.conv_c2(out_c2_interp)  # After conv_c2, 64 channels
        out_b1_conv = self.encoder.conv_b1(out_b1_interp)  # After conv_b1, 64 channels
        out_b2_conv = self.encoder.conv_b2(out_b2_interp)  # After conv_b2, 128 channels

        out_combined = torch.cat([out_c2_conv, out_b1_conv, out_b2_conv], dim=1)
        # print(out_combined.size()) # torch.Size([BS, 256, 37, 37])
        combined_L, combined_H = self.wtfd_cat(out_combined)
        DA_fusion = self.DA_fusion(combined_L, combined_H)

        
        cmf = self.cmf(out_b2, DA_fusion)
        
        out_b3 = self.encoder.block3(cmf) # torch.Size([32, 728, 19, 19])
        # out_b3 = self.encoder.block3(out_b2) # torch.Size([32, 728, 19, 19])
        embedding_b4 = self.encoder.block4(out_b3)

        norm_embed, corr = self.norm_n_corr(embedding_b4) # norm_embed.shape([32, 728, 1, 1]);corr.shape torch.Size([32, 32])
        self.loss_inputs['contra'].append(corr)

        out = self.dropout(embedding_b4)
        out = self.decoder1(out) # torch.Size([32, 256, 38, 38])
        out_d2 = self.decoder2(out) # # torch.Size([32, 256, 38, 38])

        '''cat'''
        # out_b3_dual_resized = F.interpolate(out_b3_dual, size=(37, 37), mode='bilinear', align_corners=False)
        # cat_d2 = torch.cat([out_b3_dual_resized, out_b2], dim=1)
        # conv1x1 = nn.Conv2d(cat_d2.size(1), 256, kernel_size=1, stride=1, padding=0)
        # cat_d2_adjusted = conv1x1(cat_d2)
        # # 解码器级别分解
        # out_d3 = self.decoder3(out_d2)
        # low_d4, high_d4 = self.wtfd_decoder(out_d3)
        # out_d4 = low_d4 + high_d4

        # out = self.decoder3(out_d2) # torch.Size([32, 128, 76, 76])
        # out_d4 = self.decoder4(out) # torch.Size([32, 128, 76, 76])

        # norm_embed, corr = self.norm_n_corr(out_d4) # norm_embed.shape([32, 128, 1, 1])
        # self.loss_inputs['contra'].append(corr)

        # out = self.decoder5(out_d4) # out.Size([32, 64, 152, 152])
        # pred = self.decoder6(out) # pred.Size([32, 3, 152, 152]); pred   

        norm_embed, corr = self.norm_n_corr(out_d2) # norm_embed.shape([32, 256, 1, 1]); 
        self.loss_inputs['contra'].append(corr)

        out = self.decoder3(out_d2) # torch.Size([32, 128, 76, 76])
        out_d4 = self.decoder4(out) # torch.Size([32, 128, 76, 76])

        norm_embed, corr = self.norm_n_corr(out_d4) # norm_embed.shape([32, 128, 1, 1])
        self.loss_inputs['contra'].append(corr)

        out = self.decoder5(out_d4) # out.Size([32, 64, 152, 152])
        pred = self.decoder6(out) # pred.Size([32, 3, 152, 152]); pred

        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True) # recons_x.Size([32, 3, 299, 299])

        self.loss_inputs['recons'].append(recons_x)
        ### encoder 出来
        embedding = self.encoder.block5(embedding_b4) # embedding.Size([32, 728, 19, 19])
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)
        # fusion = self.reasoning(embedding, out_d2, out_d4) + embedding # 多尺度融合
        embedding = self.encoder.block8(embedding)                 
        # embedding = self.encoder.block8(fusion)
        # img_att = self.attention(x, recons_x, embedding)

        embedding = self.encoder.block9(embedding)
        # embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding) # torch.Size([32, 728, 19, 19])【=
        embedding = self.encoder.block12(embedding) # torch.Size([32, 1024, 10, 10])
        embedding = self.encoder.conv3(embedding) # torch.Size([32, 1536, 10, 10])
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding) # torch.Size([32, 2048, 10, 10])
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze() # torch.Size([32, 2048])
        out = self.dropout(embedding) # torch.Size([32, 2048]); 2048 

        return self.fc(out)