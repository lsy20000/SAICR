import torch
from torch import nn
from torch.nn import functional as F
from .bert_encoder import MultiStageBertModel
from .swin_encoder import MultiStageSwinTransformer
from timm.models.layers import trunc_normal_
from .BIDAmodel import SCAF
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

class V_FusionGate(nn.Module):
    def __init__(self, channel, drop=0.):
        super().__init__()
        # 卷积分支
        self.fc1 = nn.Linear(channel, channel, bias=False)
        self.dwconv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True, groups=channel)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(channel, channel, bias=False)
        # SE分支
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(channel, channel, bias=False)
        self.act2 = nn.ReLU()
        self.fc4 = nn.Linear(channel, channel, bias=False)
        # 映射
        self.out = nn.Tanh()
        # self.out = nn.Sigmoid()

    def forward(self, x, H, W):
        temp_x = x
        # 卷积分支
        x = self.fc1(x)
        
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.act1(x)
        x = self.fc2(x)
        
        # SE分支
        temp_x = temp_x.transpose(1, 2).view(B, C, H, W).contiguous()# [B,C,H,W]
        temp_x = self.avgpool(temp_x)
        temp_x = temp_x.flatten(2).transpose(1, 2)

        temp_x = self.fc3(temp_x)
        temp_x = self.act2(temp_x)
        temp_x = self.fc4(temp_x)

        x = x + temp_x

        x = self.out(x)
        return x

class L_FusionGate(nn.Module):
    def __init__(self, channel, drop=0.):
        super().__init__()
        # 卷积分支
        self.fc1 = nn.Linear(channel, channel, bias=False)
        self.dwconv = nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True, groups=channel)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(channel, channel, bias=False)
        # SE分支
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc3 = nn.Linear(channel, channel, bias=False)
        self.act2 = nn.ReLU()
        self.fc4 = nn.Linear(channel, channel, bias=False)
        # 映射
        self.out = nn.Tanh()
        # self.out = nn.Sigmoid()

    def forward(self, x):
        temp_x = x
        # 卷积分支
        x = self.fc1(x)
        
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous() # [B,C,L]
        x = self.dwconv(x) # [B,C,L]
        x = x.transpose(1, 2).contiguous()

        x = self.act1(x)
        x = self.fc2(x)
        
        # SE分支
        temp_x = temp_x.transpose(1, 2).contiguous() # [B,C,L]
        temp_x = self.avgpool(temp_x) # [B,C,1]
        temp_x = temp_x.transpose(1, 2).contiguous() # [B,1,C]

        temp_x = self.fc3(temp_x)
        temp_x = self.act2(temp_x)
        temp_x = self.fc4(temp_x)

        x = x + temp_x

        x = self.out(x)
        return x

class PromptEncoder(nn.Module):

    def __init__(self, args, channels, bert_out_layers=[3, 6, 9, 12], **kwargs):
        super().__init__()
        assert len(bert_out_layers) == 4 and len(channels) == 4, 'Only 4-stage index is supported!'
        self.bert_out_layers = bert_out_layers
        self.n_stages = 4
        self.text_dim = 768
        self.vis_dim = max(channels)
        self.vis_encoder = MultiStageSwinTransformer(**kwargs)
        self.lang_encoder = MultiStageBertModel.from_pretrained(args.ck_bert)
        self.lang_encoder.pooler = None

        self.fusion = nn.ModuleList([
            SCAF(channels[i],
                     channels[i],
                     768,
                     channels[i],
                     channels[i],
                     num_heads=1,
                     dropout=0.0
            )
            for i in range(self.n_stages)
        ])
        
        self.res_gate = nn.ModuleList([
            V_FusionGate(channels[i])
            for i in range(self.n_stages)
        ])
        self.lang_res_gate = nn.ModuleList([
            L_FusionGate(768)
            for i in range(self.n_stages)
        ])


    def forward(self, x, text, l_mask):
        '''
            Args:
                x: [B, C, H, W]
                text: [B, N_l]
                l_mask:[B, N_l]
            Returns:
                vis_outs (list): multi-level visual features
                l_feats: [B, N_l, 768]
        '''
        CL_lang_outs = []
        CL_vis_outs = []
        # Vis encoding
        vis_outs = []
        l_feats, extended_l_mask = self.lang_encoder.forward_embeddings(text, attention_mask=l_mask)
        #l_feats: [B, N_l, 768]

        temp_l_mask = l_mask.unsqueeze(dim=-1) 
        #temp_l_mask: [B, N_l, 1]
        count = 0
        v_feats, Wh, Ww = self.vis_encoder.forward_embeddings(x)
        for stage in range(self.n_stages):
            v_feats, Wh, Ww = self.vis_encoder.forward_stages(v_feats, Wh, Ww, stage)
            l_feats = self.lang_encoder.forward_stages(l_feats, count, self.bert_out_layers[stage],extended_l_mask)
            count = self.bert_out_layers[stage]

            CL_lang_outs.append(l_feats[:,0,:])# [B, N_l, 768]
            CL_vis_outs.append(v_feats)
            # input v_feats need [B,H*W,dim]  
            # input l_feats need [B,l_in_channels,n_l]  
            v_residual,l_residual = self.fusion[stage](v_feats, l_feats.permute(0, 2, 1), temp_l_mask)
            # l_residual [B, N_l, 768]

            # 门控
            v_feats = v_feats + (self.res_gate[stage](v_residual, Wh, Ww) * v_residual)
            l_feats = l_feats + (self.lang_res_gate[stage](l_residual) * l_residual)
            
            vis_outs.append(self.vis_encoder.forward_norms(v_feats, Wh, Ww, stage)) # collect visual features before fusion
            v_feats, Wh, Ww = self.vis_encoder.forward_downs(v_feats, Wh, Ww, stage) # downsample
        
        return vis_outs, l_feats, l_mask, CL_vis_outs,CL_lang_outs






