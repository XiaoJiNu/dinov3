#!/usr/bin/env python3
"""
DINOv3 零样本语义分割
=====================
使用DINOTxt模型根据文本提示进行开放词汇分割

功能特点：
- 支持任意文本描述的物体分割
- 无需预训练特定类别
- 支持批量文本提示
- 提供多种可视化选项
- 支持多尺度推理
"""

import sys
import os
from datetime import datetime
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# 添加DINOv3仓库路径
REPO_DIR = "/vepfs-perception-b/chengdu/yr/code/large_models/dinov3"
sys.path.append(REPO_DIR)

# 设置torch.hub缓存目录
TORCH_CACHE_DIR = "/vepfs-perception-b/chengdu/yr/cache/torch"
os.environ['TORCH_HOME'] = TORCH_CACHE_DIR
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)

# 导入必要的库
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import requests

def download_sample_image(url: str, save_path: str) -> bool:
    """下载示例图像"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"下载图像失败: {e}")
        return False

def load_dinotxt_model(weights_path: str = None, backbone_weights_path: str = None):
    """
    加载DINOTxt模型和tokenizer
    
    Args:
        weights_path: DINOTxt权重文件路径
        backbone_weights_path: 骨干网络权重文件路径
    
    Returns:
        model, tokenizer
    """
    print("正在加载DINOTxt模型...")
    
    try:
        # 默认权重路径
        if weights_path is None:
            weights_path = "/vepfs-perception-b/chengdu/yr/code/large_models/dinov3_weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        
        if backbone_weights_path is None:
            backbone_weights_path = "/vepfs-perception-b/chengdu/yr/code/large_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        
        # 加载模型
        model, tokenizer = torch.hub.load(
            REPO_DIR,
            'dinov3_vitl16_dinotxt_tet1280d20h24l',
            source='local',
            weights=weights_path,
            backbone_weights=backbone_weights_path,
            trust_repo=True
        )
        
        print("✓ DINOTxt模型加载成功")
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("请确保：")
        print("1. DINOTxt权重文件路径正确")
        print("2. 骨干网络权重文件路径正确")
        print("3. 有足够的内存和显存")
        raise

def preprocess_image(image_path: str, size: int = 518) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    预处理输入图像
    
    Args:
        image_path: 图像路径
        size: 输入尺寸（推荐518，是14的倍数）
    
    Returns:
        image_tensor, original_image, original_size
    """
    # 加载图像
    if isinstance(image_path, str):
        original_image = Image.open(image_path).convert('RGB')
    else:
        original_image = image_path.convert('RGB')
    
    original_size = original_image.size  # (W, H)
    
    # 预处理transform
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(original_image).unsqueeze(0)
    return image_tensor, original_image, original_size

def zero_shot_segmentation(
    model,
    tokenizer,
    image_tensor: torch.Tensor,
    text_prompts: List[str],
    threshold: float = 0.3,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    执行零样本分割
    
    Args:
        model: DINOTxt模型
        tokenizer: 文本tokenizer
        image_tensor: 预处理后的图像张量 [1, 3, H, W]
        text_prompts: 文本提示列表
        threshold: 分割阈值
        device: 计算设备
    
    Returns:
        combined_mask: 合并的分割掩码 [H, W]
        similarity_maps: 原始相似度图 [num_prompts, H, W]
        individual_masks: 各个提示的分割掩码列表
    """
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    
    print(f"执行零样本分割，文本提示: {text_prompts}")
    
    all_similarities = []
    individual_masks = []
    
    with torch.no_grad():
        # 获取图像特征（含patch tokens）
        image_features, patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(
            image_tensor, normalize=True
        )
        
        print(f"图像patch tokens形状: {patch_tokens.shape}")
        
        # 处理每个文本提示
        for prompt in text_prompts:
            print(f"  处理提示: '{prompt}'")
            
            # 编码文本
            text_tokens = tokenizer.encode(prompt)
            if isinstance(text_tokens, list):
                text_tokens = torch.tensor(text_tokens)
            text_tokens = text_tokens.unsqueeze(0).to(device)
            
            # 获取文本特征
            text_features = model.encode_text(text_tokens, normalize=True)  # [1, embed_dim]
            
            # 计算patch级别的相似度
            # patch_tokens: [1, num_patches, embed_dim]
            # text_features: [1, embed_dim]
            similarity = torch.matmul(patch_tokens, text_features.T).squeeze()  # [num_patches]
            
            # 重塑为空间形状
            # 对于518x518输入，ViT-L/16产生37x37个patches
            patch_size = int(np.sqrt(similarity.shape[0]))
            similarity_2d = similarity.view(patch_size, patch_size)  # [37, 37]
            
            # 上采样到输入图像尺寸
            similarity_upsampled = F.interpolate(
                similarity_2d.unsqueeze(0).unsqueeze(0),
                size=(image_tensor.shape[2], image_tensor.shape[3]),
                mode='bilinear',
                align_corners=False
            ).squeeze()  # [518, 518]
            
            all_similarities.append(similarity_upsampled)
            
            # 生成二值掩码
            binary_mask = (similarity_upsampled > threshold).float()
            individual_masks.append(binary_mask)
    
    # 将所有相似度图堆叠
    similarity_maps = torch.stack(all_similarities)  # [num_prompts, H, W]
    
    # 合并所有掩码（取最大值）
    combined_mask = torch.stack(individual_masks).max(dim=0)[0]
    
    return combined_mask, similarity_maps, individual_masks

def resize_to_original(mask: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
    """将掩码调整到原始图像尺寸"""
    mask_resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(original_size[1], original_size[0]),  # (H, W)
        mode='nearest'
    ).squeeze().cpu().numpy()
    
    return mask_resized

def create_colored_mask(mask: np.ndarray, color_idx: int = 0) -> np.ndarray:
    """创建彩色掩码"""
    colors = [
        [255, 0, 0],      # 红色
        [0, 255, 0],      # 绿色  
        [0, 0, 255],      # 蓝色
        [255, 255, 0],    # 黄色
        [255, 0, 255],    # 品红
        [0, 255, 255],    # 青色
        [255, 128, 0],    # 橙色
        [128, 0, 255],    # 紫色
        [255, 192, 203],  # 粉色
        [173, 255, 47],   # 黄绿色
    ]
    
    color = colors[color_idx % len(colors)]
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    
    return colored_mask

def visualize_results(
    original_image: Image.Image,
    text_prompts: List[str],
    similarity_maps: torch.Tensor,
    individual_masks: List[torch.Tensor],
    combined_mask: torch.Tensor,
    original_size: Tuple[int, int],
    output_dir: str,
    timestamp: str
):
    """
    可视化和保存分割结果
    
    Args:
        original_image: 原始图像
        text_prompts: 文本提示列表
        similarity_maps: 相似度图
        individual_masks: 单独的掩码
        combined_mask: 合并掩码
        original_size: 原始图像尺寸
        output_dir: 输出目录
        timestamp: 时间戳
    """
    print("正在生成可视化结果...")
    
    # 调整所有掩码到原始尺寸
    combined_mask_resized = resize_to_original(combined_mask, original_size)
    
    # 1. 保存单独的相似度热图
    for i, (prompt, sim_map) in enumerate(zip(text_prompts, similarity_maps)):
        plt.figure(figsize=(12, 4))
        
        # 原始图像
        plt.subplot(131)
        plt.imshow(original_image)
        plt.title('原始图像')
        plt.axis('off')
        
        # 相似度热图
        plt.subplot(132)
        sim_resized = resize_to_original(sim_map, original_size)
        im = plt.imshow(sim_resized, cmap='hot', alpha=0.8)
        plt.title(f"相似度热图: '{prompt}'")
        plt.colorbar(im, shrink=0.6)
        plt.axis('off')
        
        # 分割掩码
        plt.subplot(133)
        mask_resized = resize_to_original(individual_masks[i], original_size)
        plt.imshow(original_image, alpha=0.7)
        plt.imshow(mask_resized, alpha=0.5, cmap='Reds')
        plt.title(f"分割结果: '{prompt}'")
        plt.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"similarity_{prompt.replace(' ', '_')}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存相似度图: {save_path}")
    
    # 2. 保存多类别合并结果
    plt.figure(figsize=(15, 10))
    
    # 计算子图布局
    n_prompts = len(text_prompts)
    cols = min(3, n_prompts + 1)
    rows = (n_prompts + 2) // cols
    
    # 原始图像
    plt.subplot(rows, cols, 1)
    plt.imshow(original_image)
    plt.title('原始图像', fontsize=12)
    plt.axis('off')
    
    # 各个类别的分割结果
    overlay_image = np.array(original_image)
    for i, prompt in enumerate(text_prompts):
        plt.subplot(rows, cols, i + 2)
        mask_resized = resize_to_original(individual_masks[i], original_size)
        
        # 创建叠加图像
        colored_mask = create_colored_mask(mask_resized, i)
        overlay = overlay_image.copy()
        mask_bool = mask_resized > 0.5
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + colored_mask[mask_bool] * 0.5
        
        plt.imshow(overlay)
        plt.title(f"'{prompt}'", fontsize=10)
        plt.axis('off')
    
    # 合并结果
    if len(text_prompts) > 1:
        plt.subplot(rows, cols, n_prompts + 2)
        overlay_combined = np.array(original_image)
        
        for i, prompt in enumerate(text_prompts):
            mask_resized = resize_to_original(individual_masks[i], original_size)
            colored_mask = create_colored_mask(mask_resized, i)
            mask_bool = mask_resized > 0.5
            overlay_combined[mask_bool] = overlay_combined[mask_bool] * 0.7 + colored_mask[mask_bool] * 0.3
        
        plt.imshow(overlay_combined)
        plt.title('合并结果', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"zero_shot_segmentation_{timestamp}.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存合并结果: {combined_path}")
    
    # 3. 保存掩码文件
    for i, prompt in enumerate(text_prompts):
        mask_resized = resize_to_original(individual_masks[i], original_size)
        mask_img = Image.fromarray((mask_resized * 255).astype(np.uint8), mode='L')
        mask_path = os.path.join(output_dir, f"mask_{prompt.replace(' ', '_')}_{timestamp}.png")
        mask_img.save(mask_path)
        print(f"  ✓ 保存掩码文件: {mask_path}")

def main():
    parser = argparse.ArgumentParser(description="DINOv3零样本语义分割")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--prompts", type=str, nargs='+', 
                       default=["cat", "person", "car"], 
                       help="文本提示列表")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="分割阈值 (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="./zero_shot_outputs",
                       help="输出目录")
    parser.add_argument("--weights", type=str, 
                       help="DINOTxt权重文件路径")
    parser.add_argument("--backbone_weights", type=str,
                       help="骨干网络权重文件路径")
    parser.add_argument("--size", type=int, default=518,
                       help="输入图像尺寸")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 处理输入图像
    if args.image is None:
        # 下载示例图像
        sample_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        sample_path = os.path.join(args.output_dir, "sample_image.jpg")
        print(f"未指定输入图像，正在下载示例图像...")
        
        if download_sample_image(sample_url, sample_path):
            args.image = sample_path
            print(f"✓ 示例图像已下载: {sample_path}")
        else:
            print("✗ 示例图像下载失败")
            return
    
    if not os.path.exists(args.image):
        print(f"✗ 图像文件不存在: {args.image}")
        return
    
    print("=" * 60)
    print("DINOv3 零样本语义分割")
    print("=" * 60)
    print(f"输入图像: {args.image}")
    print(f"文本提示: {args.prompts}")
    print(f"分割阈值: {args.threshold}")
    print(f"输出目录: {args.output_dir}")
    
    try:
        # 1. 加载模型
        model, tokenizer = load_dinotxt_model(args.weights, args.backbone_weights)
        
        # 2. 预处理图像
        print("\n正在预处理图像...")
        image_tensor, original_image, original_size = preprocess_image(args.image, args.size)
        print(f"原始图像尺寸: {original_size}")
        print(f"预处理后尺寸: {image_tensor.shape}")
        
        # 3. 执行零样本分割
        print("\n执行零样本分割...")
        combined_mask, similarity_maps, individual_masks = zero_shot_segmentation(
            model, tokenizer, image_tensor, args.prompts, args.threshold, device
        )
        
        # 4. 可视化和保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n保存结果...")
        
        visualize_results(
            original_image, args.prompts, similarity_maps, individual_masks,
            combined_mask, original_size, args.output_dir, timestamp
        )
        
        print("\n" + "=" * 60)
        print("零样本分割完成！")
        print(f"结果已保存到: {args.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 执行过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()