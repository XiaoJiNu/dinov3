# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """
    Gram锚定损失函数的实现
    
    Gram锚定是DINOv3的核心创新，用于解决大规模SSL训练中的密集特征退化问题。
    该损失不直接约束特征向量本身，而是约束特征之间的相对相似性结构（通过Gram矩阵表示）。
    """

    def __init__(
        self,
        apply_norm=True,  # 是否对特征进行归一化
        img_level=True,   # 是否在图像级别计算Gram矩阵
        remove_neg=True,  # 是否移除负相似度值
        remove_only_teacher_neg=False,  # 是否只移除教师的负相似度值
    ):
        super().__init__()

        # 损失函数：使用MSE损失来比较Gram矩阵
        self.mse_loss = torch.nn.MSELoss()

        # 参数配置
        self.apply_norm = apply_norm  # 是否应用L2归一化
        self.remove_neg = remove_neg  # 是否移除所有负相似度
        self.remove_only_teacher_neg = remove_only_teacher_neg  # 是否只移除教师的负相似度

        # 确保两个移除负值的选项不会同时为True
        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):
        """
        计算学生特征和目标特征的Gram矩阵之间的MSE损失
        
        Gram矩阵定义为 X @ X^T，其中X是特征矩阵。这个矩阵编码了所有特征向量
        之间的两两相似度关系，是特征空间结构的紧凑表示。

        Args:
            output_feats: 学生模型的特征 (B, N, dim) 或 (B*N, dim) 当img_level=False时
            target_feats: 目标特征（Gram教师）(B, N, dim) 或 (B*N, dim) 当img_level=False时  
            img_level: bool, 如果为True则在图像级别计算Gram矩阵，否则在整个批次上计算
        Returns:
            loss: 标量损失值
        """

        # 在图像级别时，输入张量维度应该是 (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # 转换为float类型以确保数值稳定性
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # 处理目标特征（Gram教师特征）
        if self.apply_norm:
            # L2归一化：将特征归一化到单位球面上，这样相似度就是余弦相似度
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            # 将 (B, N, D) 展平为 (B*N, D) 以在批次级别计算
            target_feats = target_feats.flatten(0, 1)

        # 计算目标特征的Gram矩阵：target_sim[i,j] = target_feats[i] · target_feats[j]
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # 处理学生特征
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # 将 (B, N, D) 展平为 (B*N, D)
            output_feats = output_feats.flatten(0, 1)

        # 计算学生特征的Gram矩阵
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        # 处理负相似度值的策略
        if self.remove_neg:
            # 将所有负相似度值设为0（适用于某些情况下的数值稳定性）
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # 只移除教师的负相似度值，并在相应位置同时移除学生的负值
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        # 返回两个Gram矩阵之间的MSE损失
        # 这个损失衡量学生特征和教师特征的相对相似性结构有多接近
        return self.mse_loss(student_sim, target_sim)
