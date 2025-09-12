#!/usr/bin/env python3
"""
ADE20K推理脚本 - 基于ade20k_train.py训练的模型进行语义分割推理
支持GPU推理并保存可视化结果
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
import json

# 模型配置 - 与训练脚本保持一致
REPO_DIR = "/home/yr/yr/code/cv/large_models/dinov3_all/dinov3"
WEIGHTS_PATH = "/home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
LOCAL_MODEL_PATH = "/home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"  
N_CLASSES = 150  # ADE20K有150个类别

# ADE20K颜色映射 - 为每个类别分配不同颜色
def generate_ade20k_colormap(num_classes=150):
    """生成ADE20K颜色映射"""
    # 使用HSV颜色空间生成不同的颜色
    colors = []
    for i in range(num_classes):
        hue = (i * 137.508) % 360  # 使用黄金角度分割
        saturation = 0.7 + (i % 3) * 0.1  # 变化饱和度
        value = 0.8 + (i % 2) * 0.2  # 变化明度
        rgb = mcolors.hsv_to_rgb([hue/360, saturation, value])
        colors.append(rgb)
    
    # 第一个类别(背景)设为黑色
    colors[0] = [0, 0, 0]
    return np.array(colors)

class LinearSegHead(nn.Module):  
    """线性分割头 - 与训练脚本保持一致"""
    def __init__(self, in_ch, n_classes):  
        super().__init__()  
        self.proj = nn.Conv2d(in_ch, n_classes, 1)  
        self.up = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)  
    
    def forward(self, fmap): 
        return self.up(self.proj(fmap))  

def load_dinov3_model(model_path=LOCAL_MODEL_PATH):
    """加载本地DINOv3模型 - 与训练脚本保持一致"""
    try:
        # 方法1: 使用正确的torch.hub.load方式加载本地模型
        print(f"尝试从DINOv3 repo加载模型: {REPO_DIR}")
        print(f"使用权重文件: {WEIGHTS_PATH}")
        model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH, trust_repo=True)
        print("✓ 成功从本地路径加载DINOv3模型")
        return model, None  # 本地模型不需要processor
    except Exception as e:
        print(f"本地torch.hub加载失败: {e}")
        try:
            # 方法2: 回退到transformers
            print("回退到transformers...")
            proc = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=True)
            print("✓ 使用transformers本地加载模型")
            return model, proc
        except Exception as e2:
            print(f"transformers本地加载失败: {e2}")
            try:
                # 方法3: 在线transformers (最后回退)
                print("尝试在线下载transformers...")
                proc = AutoImageProcessor.from_pretrained(MODEL_ID)
                model = AutoModel.from_pretrained(MODEL_ID, device_map="auto")
                print("✓ 使用transformers在线加载模型")
                return model, proc
            except Exception as e3:
                raise RuntimeError(f"所有模型加载方法都失败。错误: {e}, {e2}, {e3}")

def extract_fmap_local(model, image_tensor):
    """从本地DINOv3模型提取特征图 - 与训练脚本保持一致"""
    # image_tensor已经预处理过，只需要确保batch维度和设备
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # 确保图像在正确的设备上
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 对于torch.hub加载的模型，使用forward_features
        if hasattr(model, 'forward_features'):
            # forward_features返回字典，包含'x_norm_patchtokens'等键
            features_dict = model.forward_features(image_tensor)
            # 获取patch tokens (排除CLS token和register tokens)
            patch_features = features_dict['x_norm_patchtokens']  # [B, N_patches, C]
            B, N, C = patch_features.shape
            H = W = int(N ** 0.5)
            return patch_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            # 如果是transformers模型，使用原来的方法
            raise AttributeError("Model doesn't have forward_features method")

def extract_fmap(model, proc, image):  
    """从DINOv3模型提取特征图 - transformers版本"""
    inputs = proc(images=image, return_tensors="pt").to(model.device)  
    with torch.inference_mode():  
        out = model(**inputs)  
    num_regs = model.config.num_register_tokens  
    grid = out.last_hidden_state[:, 1 + num_regs:, :]  # 丢弃CLS+寄存器
    B, N, C = grid.shape  
    H = W = int(N ** 0.5)  
    return grid.reshape(B, H, W, C).permute(0, 3, 1, 2)  

def preprocess_image(image_path, image_size=512):
    """预处理输入图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 预处理transform - 与训练脚本保持一致
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    return image_tensor, image, original_size

def postprocess_prediction(pred_mask, original_size, image_size):
    """后处理预测结果，调整到原始图像尺寸"""
    # pred_mask: (H, W)
    pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
    # 调整到原始尺寸
    pred_resized = pred_pil.resize(original_size, Image.NEAREST)
    return np.array(pred_resized)

def colorize_segmentation(seg_mask, colormap):
    """将分割掩码转换为彩色图像"""
    h, w = seg_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(colormap)):
        mask = seg_mask == class_id
        colored[mask] = (colormap[class_id] * 255).astype(np.uint8)
    
    return colored

def overlay_segmentation(original_image, seg_colored, alpha=0.6):
    """将分割结果叠加到原始图像上"""
    original_array = np.array(original_image)
    blended = (1 - alpha) * original_array + alpha * seg_colored
    return blended.astype(np.uint8)

def save_visualization(original_image, seg_mask, seg_colored, overlay, output_path):
    """保存可视化结果"""
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 分割掩码(灰度)
    axes[0, 1].imshow(seg_mask, cmap='tab20')
    axes[0, 1].set_title('Segmentation Mask')
    axes[0, 1].axis('off')
    
    # 彩色分割
    axes[1, 0].imshow(seg_colored)
    axes[1, 0].set_title('Colored Segmentation')
    axes[1, 0].axis('off')
    
    # 叠加结果
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def inference_single_image(model, head, proc, image_path, output_dir, colormap, image_size=512):
    """对单张图像进行推理"""
    print(f"处理图像: {image_path}")
    
    # 预处理
    image_tensor, original_image, original_size = preprocess_image(image_path, image_size)
    
    # 特征提取
    device = next(model.parameters()).device
    
    if proc is not None:
        # 使用transformers模型
        fmap = extract_fmap(model, proc, image_tensor)
    else:
        # 使用本地模型
        fmap = extract_fmap_local(model, image_tensor)
    
    # 推理
    with torch.no_grad():
        logits = head(fmap)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    # 后处理
    pred_resized = postprocess_prediction(pred, original_size, image_size)
    
    # 可视化
    seg_colored = colorize_segmentation(pred_resized, colormap)
    overlay = overlay_segmentation(original_image, seg_colored)
    
    # 保存结果
    image_name = Path(image_path).stem
    
    # 保存各种格式的结果
    seg_mask_path = output_dir / f"{image_name}_mask.png"
    seg_colored_path = output_dir / f"{image_name}_colored.png"
    overlay_path = output_dir / f"{image_name}_overlay.png"
    visualization_path = output_dir / f"{image_name}_visualization.png"
    
    # 保存分割掩码
    Image.fromarray(pred_resized.astype(np.uint8)).save(seg_mask_path)
    
    # 保存彩色分割
    Image.fromarray(seg_colored).save(seg_colored_path)
    
    # 保存叠加图像
    Image.fromarray(overlay).save(overlay_path)
    
    # 保存完整可视化
    save_visualization(original_image, pred_resized, seg_colored, overlay, visualization_path)
    
    print(f"结果已保存:")
    print(f"  - 分割掩码: {seg_mask_path}")
    print(f"  - 彩色分割: {seg_colored_path}")
    print(f"  - 叠加图像: {overlay_path}")
    print(f"  - 完整可视化: {visualization_path}")
    
    return pred_resized

def main():
    parser = argparse.ArgumentParser(description="ADE20K语义分割推理")
    parser.add_argument("--input", type=str, default="/home/yr/yr/data/ade20k/ade/ADEChallengeData2016/images/validation", # ADE_val_00000001.jpg
                       help="输入图像路径或目录")
    parser.add_argument("--model_path", type=str, 
                       default="./ade20k_results/epoch_0.pth",
                       help="训练好的分割头模型路径")
    parser.add_argument("--output_dir", type=str, default="./ade20k_outputs",
                       help="输出目录")
    parser.add_argument("--image_size", type=int, default=512,
                       help="输入图像尺寸")
    parser.add_argument("--dinov3_path", type=str, 
                       default=LOCAL_MODEL_PATH,
                       help="DINOv3模型路径")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载DINOv3模型
    print("加载DINOv3模型...")
    model, proc = load_dinov3_model(args.dinov3_path)
    model = model.to(device)
    model.eval()
    
    # 推断特征维度
    print("推断特征维度...")
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    if proc is not None:
        # transformers模型需要特殊处理
        # 使用一个虚拟的PIL图像来推断维度
        dummy_pil = Image.fromarray(np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8))
        dummy_fmap = extract_fmap(model, proc, dummy_pil)
    else:
        dummy_fmap = extract_fmap_local(model, dummy_input)
    
    feature_dim = dummy_fmap.shape[1]
    print(f"特征维度: {feature_dim}")
    
    # 创建并加载分割头
    head = LinearSegHead(feature_dim, N_CLASSES).to(device)
    print(f"加载分割头: {args.model_path}")
    head.load_state_dict(torch.load(args.model_path, map_location=device))
    head.eval()
    
    # 生成颜色映射
    colormap = generate_ade20k_colormap(N_CLASSES)
    
    # 处理输入
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            inference_single_image(model, head, proc, str(input_path), 
                                 output_dir, colormap, args.image_size)
        else:
            print(f"不支持的文件格式: {input_path.suffix}")
    
    elif input_path.is_dir():
        # 目录中的所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"在目录 {input_path} 中未找到支持的图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        for image_file in image_files:
            try:
                inference_single_image(model, head, proc, str(image_file), 
                                     output_dir, colormap, args.image_size)
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
    
    else:
        print(f"输入路径不存在: {input_path}")
    
    print(f"\n推理完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()