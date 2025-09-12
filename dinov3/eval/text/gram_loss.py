# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn.functional as F


def gram_loss_fn(
    backbone_patch_tokens: torch.Tensor,  # 骨干网络（教师）的补丁特征
    patch_tokens: torch.Tensor,  # 当前模型（学生）的补丁特征
    patch_sampling_rate: float = 1.0,  # 补丁采样率，用于减少计算量
    normalize: bool = True,  # 是否对特征进行归一化
) -> torch.Tensor:
    """
    文本对齐训练中使用的简化版Gram损失函数
    
    这是一个轻量级的Gram损失实现，主要用于dino.txt（文本对齐）训练中。
    通过随机采样补丁来减少计算复杂度，同时保持Gram矩阵结构约束的效果。
    
    Args:
        backbone_patch_tokens: 骨干网络的补丁特征 [batch_size, num_patches, dim]
        patch_tokens: 当前模型的补丁特征 [batch_size, num_patches, dim] 
        patch_sampling_rate: 采样率，控制使用多少比例的补丁进行计算
        normalize: 是否进行L2归一化
        
    Returns:
        gram_loss: Gram矩阵之间的MSE损失
    """
    num_patches, dim = patch_tokens.shape[1:]  # 获取补丁数量和特征维度
    
    # 随机采样补丁索引，减少计算量（特别是当补丁数量很大时）
    idx = torch.randperm(num_patches)[: int(num_patches * patch_sampling_rate)]
    
    # 根据采样索引选择补丁
    patch_tokens = patch_tokens[:, idx, :]  # [batch_size, sampled_patches, dim]
    backbone_patch_tokens = backbone_patch_tokens[:, idx, :]  # [batch_size, sampled_patches, dim]
    
    if normalize:
        # L2归一化：将特征向量归一化到单位球面，使相似度计算更稳定
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        backbone_patch_tokens = F.normalize(backbone_patch_tokens, dim=-1)
    
    # 计算并比较两个模型的Gram矩阵
    # patch_tokens @ patch_tokens.transpose(-2, -1) 计算Gram矩阵
    # 形状: [batch_size, sampled_patches, sampled_patches]
    return torch.nn.MSELoss()(
        patch_tokens @ patch_tokens.transpose(-2, -1),  # 学生模型的Gram矩阵
        backbone_patch_tokens @ backbone_patch_tokens.transpose(-2, -1),  # 教师模型的Gram矩阵
    )
