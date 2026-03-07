import torch.nn as nn
import torch
from torch.nn import functional as F
import warnings
import math

class SegCELoss(nn.Module):
    def __init__(self):
        super(SegCELoss, self).__init__()
        weight = torch.FloatTensor([0.9, 1.1]).cuda()
        self.seg_criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, targets):
        '''
            pred: [BxKxhxw]
            targets['mask']: [BxHxW]
        '''
        target = targets['mask']
        if pred.shape[-2:] != target.shape[-2:]:
            h, w = target.size(1), target.size(2)
            pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)
        seg_loss = self.seg_criterion(pred, target)
        return seg_loss


class InstanceContrastiveLoss(nn.Module):
    def __init__(self, batch_size = 8, temperature = 0.5):
        super(InstanceContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.vis_cls_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )


    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    # def forward(self, Vmask, cls):
    def forward(self, v_mask, cls, targets):
        device = v_mask.device
        # Step 1: 
        Vmask_pooled = F.normalize(self.vis_cls_proj(v_mask), dim=1)
        cls = F.normalize(self.vis_cls_proj(cls), dim=1)


        # Step 2: 拼接特征（Vmask和cls交替）
        N = 2 * self.batch_size
        z = torch.cat((Vmask_pooled, cls), dim=0)  # [2B, D]

        # Step 3: 计算相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # Step 4: 提取正样本对（Vmask[i]与cls[i]的相似度）
        sim_i_j = torch.diag(sim, self.batch_size)    # Vmask[i] vs cls[i]
        sim_j_i = torch.diag(sim, -self.batch_size)   # cls[i] vs Vmask[i]

        # Step 5: 构造正负样本logits
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)  # 掩码后的负样本

        # Step 6: 计算对比损失
        labels = torch.zeros(N).to(device).long()  # 正样本在logits第0列
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        # print('loss3: ',loss)
        return loss

class CenterLossWithLanguage(nn.Module):
    def __init__(self, class_num = 80, temperature = 1.0 ):
        super(CenterLossWithLanguage, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.vis_cls_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.class_num),
            nn.Softmax(dim=1)
        )


    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    # def forward(self, c_i, c_j):
    def forward(self, v_mask, cls, targets):
        device = v_mask.device
        
        #将Vmask从[B, D, H, W]压缩为[B, D]
        v_mask = self.vis_cls_proj(v_mask) # [B,N]
        cls = self.vis_cls_proj(cls) # [B,N]

        # 计算视觉特征的聚类分布熵
        p_i = v_mask.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        # 计算语言特征的聚类分布熵
        p_j = cls.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # 对比损失计算 --------------------------------------------
        # 转置特征矩阵：使每行代表一个聚类中心的分布
        v_mask = v_mask.t() # [N,B]
        cls = cls.t() # [N,B]
        # 拼接两个视图的特征
        N = 2 * self.class_num
        c = torch.cat((v_mask, cls), dim=0)# [2*N, B]

        # 计算所有聚类中心间的余弦相似度
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        # 提取正样本对（跨视图的相同聚类中心）
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        # 构造损失输入
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        # 交叉熵损失计算
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        # print('loss: ',loss)
        # print('ne_loss: ',ne_loss)
        # print('    ')
        return loss + ne_loss


criterion_dict = {
    'saicr': SegCELoss,
    'CenterLossWithLanguage':CenterLossWithLanguage,
    'InstanceContrastiveLoss':InstanceContrastiveLoss
}