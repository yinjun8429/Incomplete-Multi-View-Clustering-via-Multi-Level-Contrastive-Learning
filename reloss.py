import torch
import torch.nn as nn
import math
import sys


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)  # 计算余弦相似度
        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # 计算交叉熵损失

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)  # 对角线填充为0
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()  # 转换为布尔类型
        return mask

    # 对比学习损失
    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)  # 将两个张量拼接在一起（0：行 1：列）

        # sim: N * N
        sim = torch.matmul(h, h.T) / self.temperature_f  # 张量乘法

        # 输出对角线元素（0：主对角线，正数：主对角线右上角，负数：主对角线左下角）
        # 这样取对角线正好是两个视图的同一样本组成的特征对,所以是正样本
        # <hi,hj>
        sim_i_j = torch.diag(sim, self.batch_size)
        # <hj,hi>
        sim_j_i = torch.diag(sim, -self.batch_size)

        # 正样本 <hi,hj>和<hj,hi>,算两个特征对: 两倍batch_size
        # 主对角线的特征对不需要计算，因为其视图是同一个
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)  # 参数中存在负数，则表示该参数由其他参数决定

        labels = torch.zeros(N).to(positive_samples.device).long()  # device 分配设备
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    # 语义标签损失
    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        # 转置
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)  # 取对角线元素
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    def compute_joint(self, view1, view2):
        """Compute the joint probability matrix P"""

        bn, k = view1.size()
        assert (view2.size(0) == bn and view2.size(1) == k)

        p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def crossview_contrastive_Loss(self, view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        _, k = view1.size()
        p_i_j = self.compute_joint(view1, view2)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

        p_i_j[(p_i_j < EPS).data] = EPS
        p_j[(p_j < EPS).data] = EPS
        p_i[(p_i < EPS).data] = EPS

        loss = - p_i_j * (torch.log(p_i_j) \
                          - (lamb + 1) * torch.log(p_j) \
                          - (lamb + 1) * torch.log(p_i))

        loss = loss.sum()

        return loss