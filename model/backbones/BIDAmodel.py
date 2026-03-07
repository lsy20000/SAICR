import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Tuple, Union, List, Any
from torch import Tensor
import math

class SCAF(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(SCAF, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )
        
        self.lang_project = nn.Sequential(nn.Conv1d(l_in_channels,l_in_channels,1,1),
                                          nn.GELU(),
                                          nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )
        
        self.lang_project_mm = nn.Sequential(nn.Conv1d(l_in_channels,l_in_channels,1,1),
                                             nn.GELU(),
                                             nn.Dropout(dropout)
                                            )
        
    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        # l:[B,l_in_channels,N_l]
        # l_mask:(B, N_l, 1)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        lang = self.lang_project(l)

        vis_out,lang_out = self.image_lang_att(x, l, l_mask)  # vis_out: (B, H*W, dim) lang_out:(B, l_in_channels, N_l)

        vis_out = vis_out.permute(0, 2, 1)  # (B, dim, H*W)
        mm = torch.mul(vis,vis_out)
        mm = self.project_mm(mm)  # (B, dim, H*W)
        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        lang_mm = torch.mul(lang, lang_out)   # (B, l_in_channels, N_l)
        lang_mm = self.lang_project_mm(lang_mm)# (B, l_in_channels, N_l)
        lang_mm = lang_mm.permute(0, 2, 1)# (B, N_l, l_in_channels)

        return mm,lang_mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)  
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # lang作为查询时，应该将vis当作value
        self.lang_value = nn.Sequential(    
            nn.Conv1d(self.v_in_channels, self.l_in_channels, kernel_size=1,stride=1),
            nn.InstanceNorm1d(self.l_in_channels),
        ) 

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

        self.lang_W = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.l_in_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.l_in_channels) 
        )


    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        # assert not torch.isnan(l).any(), "NaN in input l"
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        # print('l_mask.shape: ',l_mask.shape)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)

        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        lang_value = self.lang_value(x)# (B, l_in_channels, H*W)

        n_l = l_mask.size(-1)
        lang_value = lang_value.reshape(B, self.num_heads, self.l_in_channels//self.num_heads, HW)# (B, num_heads, l_in_channels//self.num_heads, H*W)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)

        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)

        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        temp_l_mask = l_mask
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)
        
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product
        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions

        lang_sim_map = sim_map.permute(0,1,3,2)
        sim_map = F.softmax(sim_map, dim=-1) # + 1e-8  # (B, num_heads, h*w, N_l)

        lang_sim_map = F.softmax(lang_sim_map, dim=-1)#  (B, num_heads, N_l, h*w)

        lang_out = torch.matmul(lang_sim_map,lang_value.permute(0, 1, 3, 2))#  (B, num_heads, N_l, l_in_channels//self.num_heads)
        lang_out = lang_out.permute(0, 1, 3, 2).contiguous()#  (B, num_heads, l_in_channels//self.num_heads, N_l)
        lang_out = lang_out.reshape(B, self.l_in_channels, n_l)#  (B, l_in_channels, N_l)
        lang_out = lang_out * temp_l_mask
        lang_out = self.lang_W(lang_out)
   
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)

        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out,lang_out



