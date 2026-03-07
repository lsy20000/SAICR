import torch
from torch import nn
from torch.nn import functional as F

# for test
# class SAICR0(nn.Module):
class SAICR(nn.Module):
    def __init__(self, backbone, pixel_decoder, args, num_classes=1, criterion=None):
        super(SAICR, self).__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.num_classes = num_classes

        self.criterion = criterion
        self.base_lr = args.lr


    def params_to_optimize(self, scale_lang=0.1, scale_vis=0.1):
        # parameters to optimize
        names_frozen = list()
        names_no_decay = list()
        lang_backbone_names_no_decay = list()
        lang_backbone_params_no_decay = list()
        lang_backbone_params_decay = list()
        backbone_names_no_decay = list()
        backbone_params_no_decay = list()
        backbone_params_decay = list()
        params_no_decay = list()
        params_decay = list()
        for name, m in self.named_parameters():
            if m.requires_grad:
                if 'backbone' in name:
                    # Language backbone
                    if 'lang_encoder' in name:
                        if 'Norm' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        elif 'embeddings' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        else:
                            lang_backbone_params_decay.append(m)
                    # Visual backbone
                    elif 'vis_encoder' in name:
                        if 'norm' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'position_embeddings' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        else:
                            backbone_params_decay.append(m)
                    # Others
                    elif 'lang_prompts' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
                else:
                    if 'norm' in name or 'Norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'prompt' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
            else:
                names_frozen.append(name)

        params_to_optimize = [
            {'params': lang_backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_lang * self.base_lr},
            {'params': lang_backbone_params_decay, 'lr': scale_lang * self.base_lr},
            {'params': backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_vis * self.base_lr},
            {'params': backbone_params_decay, 'lr': scale_vis * self.base_lr},
            {'params': params_no_decay, 'weight_decay': 0.0, 'lr': self.base_lr},
            {'params': params_decay, 'lr': self.base_lr},
        ]
        print('scale_lang_backbone: ', scale_lang)
        print('scale_vis_backbone: ', scale_vis)
        print('LANG BACKBONE NO DECAY params: ', lang_backbone_names_no_decay)
        print('BACKBONE NO DECAY params: ', backbone_names_no_decay)
        print('NO DECAY params: ', names_no_decay)
        print('FROZEN params: ', names_frozen)
        return params_to_optimize

    def forward(self, x, text, l_mask, resize_output=True, targets=None, return_probs=False, return_attn=False, all_text = None, all_l_mask = None):
        '''
            Input:
                x       [BxCxHxW]
                text    [BxN_l]
                l_mask  [BxN_l]
        '''
        input_shape = x.shape[-2:] # H,W
        lang_len = l_mask.shape[1]
 
        # Multi-modal encoding
        # outs : return vis_outs, l_feats, l_mask
        if all_text == None:
            outs = self.backbone(x, text, l_mask) #vis_outs[-1]: [B, C, H, W] l_feats: [B, N_l, 768]
            vis_outs = outs[0]
            l_feats = outs[1]

        # VL pixel decoder
        l_feats = l_feats[:,:lang_len] # [B, N_l, 768]

        x, lang_cls, v4_feature = self.pixel_decoder(vis_outs, l_feats, l_mask)
        
        if resize_output:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            if return_attn:
                attns = [F.interpolate(attn, size=input_shape, mode='bilinear', align_corners=True) for attn in attns]
                attns = [attn.reshape(x.shape[0], self.pixel_decoder.num_enc_layers, -1, input_shape[0], input_shape[1]) for attn in attns] # [B, N_layer, N_l, H, W]
        if x.shape[1] == 1:
            if not return_probs:
                x = x.sigmoid()
                x = (x >= 0.5) * 1
        else:
            if not return_probs:
                x = torch.argmax(x, dim=1, keepdim=True)
        if return_attn:
            return x, attns
        return x

# for train
class SAICR1(nn.Module):
# class SAICR(nn.Module):
    def __init__(self, backbone, pixel_decoder, args, num_classes=1, criterion=None):
        super(SAICR, self).__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.num_classes = num_classes

        self.criterion = criterion
        self.base_lr = args.lr

        self.para = nn.Parameter(torch.tensor(0.0)) 
        
        self.center_loss = self.criterion[1]().cuda()
        self.Instance_Loss = self.criterion[2]().cuda()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def params_to_optimize(self, scale_lang=0.1, scale_vis=0.1):
        # parameters to optimize
        names_frozen = list()
        names_no_decay = list()
        lang_backbone_names_no_decay = list()
        lang_backbone_params_no_decay = list()
        lang_backbone_params_decay = list()
        backbone_names_no_decay = list()
        backbone_params_no_decay = list()
        backbone_params_decay = list()
        params_no_decay = list()
        params_decay = list()
        for name, m in self.named_parameters():
            if m.requires_grad:
                if 'backbone' in name:
                    # Language backbone
                    if 'lang_encoder' in name:
                        if 'Norm' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        elif 'embeddings' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        else:
                            lang_backbone_params_decay.append(m)
                    # Visual backbone
                    elif 'vis_encoder' in name:
                        if 'norm' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'position_embeddings' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        else:
                            backbone_params_decay.append(m)
                    # Others
                    elif 'lang_prompts' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
                else:
                    if 'norm' in name or 'Norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'prompt' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
            else:
                names_frozen.append(name)

        params_to_optimize = [
            {'params': lang_backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_lang * self.base_lr},
            {'params': lang_backbone_params_decay, 'lr': scale_lang * self.base_lr},
            {'params': backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_vis * self.base_lr},
            {'params': backbone_params_decay, 'lr': scale_vis * self.base_lr},
            {'params': params_no_decay, 'weight_decay': 0.0, 'lr': self.base_lr},
            {'params': params_decay, 'lr': self.base_lr},
        ]
        print('scale_lang_backbone: ', scale_lang)
        print('scale_vis_backbone: ', scale_vis)
        print('LANG BACKBONE NO DECAY params: ', lang_backbone_names_no_decay)
        print('BACKBONE NO DECAY params: ', backbone_names_no_decay)
        print('NO DECAY params: ', names_no_decay)
        print('FROZEN params: ', names_frozen)
        return params_to_optimize

    def forward(self, x, text, l_mask, resize_output=True, targets=None, return_probs=False, return_attn=False, all_text = None, all_l_mask = None):
        '''
            Input:
                x       [BxCxHxW]
                text    [BxN_l]
                l_mask  [BxN_l]
        '''
        input_shape = x.shape[-2:] # H,W
        lang_len = l_mask.shape[1]
 
        # Multi-modal encoding
        # outs : return vis_outs, l_feats, l_mask
        if all_text == None:
            outs = self.backbone(x, text, l_mask) #vis_outs[-1]: [B, C, H, W] l_feats: [B, N_l, 768]
            vis_outs = outs[0]
            l_feats = outs[1]

        # VL pixel decoder
        l_feats = l_feats[:,:lang_len] # [B, N_l, 768]
        cls = l_feats[:,0]

        x, lang_cls, v4_feature= self.pixel_decoder(vis_outs, l_feats, l_mask)

        if self.training:
            if self.criterion is not None:
                #SegCELoss
                loss_function1 = self.criterion[0]()
                loss1 = loss_function1(x, targets)
                # print(loss1)

                #Center_Loss
                v4_feature = self.avgpool(v4_feature)
                v4_feature = torch.flatten(v4_feature, 1) 
                loss2 = self.center_loss(v4_feature, lang_cls, targets)
                # print(loss2)

                #Instance_Loss
                loss3 = self.Instance_Loss(v4_feature, lang_cls, targets)
                # print(loss3)

                loss_dict = {'total_loss1':loss1 , 'total_loss2':loss2*0.01 , 'total_loss3':loss3*0.01}
                return loss_dict
            


        if resize_output:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            if return_attn:
                attns = [F.interpolate(attn, size=input_shape, mode='bilinear', align_corners=True) for attn in attns]
                attns = [attn.reshape(x.shape[0], self.pixel_decoder.num_enc_layers, -1, input_shape[0], input_shape[1]) for attn in attns] # [B, N_layer, N_l, H, W]
        if x.shape[1] == 1:
            if not return_probs:
                x = x.sigmoid()
                x = (x >= 0.5) * 1
        else:
            if not return_probs:
                x = torch.argmax(x, dim=1, keepdim=True)
        if return_attn:
            return x, attns
        return x


