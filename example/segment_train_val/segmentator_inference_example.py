#!/usr/bin/env python3
"""
DINOv3 语义分割推理示例
======================
从官方README提取的语义分割代码，添加了详细的中文注释

功能：使用预训练的DINOv3-ViT-7B/16模型对输入图像进行语义分割
数据集：模型在ADE20K数据集上训练，支持150个语义类别
"""

import sys
import os
from datetime import datetime
import numpy as np

# ===== 重要配置 =====
# DINOv3仓库目录 - 需要根据实际情况修改
REPO_DIR = "/vepfs-perception-b/chengdu/yr/code/large_models/dinov3"

# 设置torch.hub缓存目录到有足够空间的位置
TORCH_CACHE_DIR = "/vepfs-perception-b/chengdu/yr/cache/torch"
os.environ['TORCH_HOME'] = TORCH_CACHE_DIR

# 输出目录配置
OUTPUT_DIR = "./segmentation_outputs"

# 创建必要的目录
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"设置torch.hub缓存目录: {TORCH_CACHE_DIR}")
print(f"设置输出目录: {OUTPUT_DIR}")

# 将DINOv3仓库目录添加到Python路径中，这样可以导入dinov3模块
sys.path.append(REPO_DIR)

# ===== 导入必要的库 =====
from PIL import Image  # 图像处理库
import torch  # PyTorch深度学习框架
from torchvision import transforms  # 图像预处理工具
import matplotlib.pyplot as plt  # 绘图库
from matplotlib import colormaps  # 颜色映射
from functools import partial  # 函数工具
# 导入DINOv3的分割推理函数
from dinov3.eval.segmentation.inference import make_inference

def get_img(image_path=None):
    """
    获取测试图像
    ==========
    
    Args:
        image_path (str, optional): 自定义图像路径。如果不提供，则下载示例图像
    
    Returns:
        PIL.Image: RGB格式的图像对象
    """
    if image_path and os.path.exists(image_path):
        print(f"加载本地图像: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print(f"图像加载完成，尺寸: {image.size}")
        return image
    else:
        # 下载示例图像
        import requests
        # COCO验证集中的一张图片，包含两只猫
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        
        print(f"正在下载测试图像: {url}")
        # 通过HTTP请求下载图像
        response = requests.get(url, stream=True)
        # 打开图像并转换为RGB格式（确保3通道）
        image = Image.open(response.raw).convert("RGB")
        print(f"图像下载完成，尺寸: {image.size}")
        
        return image

def make_transform(resize_size: int | list[int] = 768):
    """
    创建图像预处理变换管道
    ===================
    
    Args:
        resize_size (int): 图像缩放尺寸，默认768像素
    
    Returns:
        transforms.Compose: 图像预处理管道
    
    预处理步骤详解：
    1. ToTensor(): 将PIL图像(HWC, 0-255)转换为PyTorch张量(CHW, 0-1)
    2. Resize(): 将图像缩放到指定尺寸，保持纵横比并用插值填充
    3. Normalize(): 使用ImageNet数据集的均值和标准差进行标准化
    
    ImageNet标准化参数：
    - 均值: [0.485, 0.456, 0.406] (RGB三个通道)
    - 标准差: [0.229, 0.224, 0.225] (RGB三个通道)
    """
    # 转换为PyTorch张量
    to_tensor = transforms.ToTensor()
    
    # 缩放图像到指定尺寸，使用抗锯齿技术提高质量
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    
    # ImageNet数据集的标准化参数
    # 这些参数是在大量ImageNet图像上计算得到的统计值
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # RGB三个通道的均值
        std=(0.229, 0.224, 0.225),   # RGB三个通道的标准差
    )
    
    # 将所有变换组合成一个管道
    return transforms.Compose([to_tensor, resize, normalize])

def save_segmentation_results(original_img, segmentation_result, output_dir, timestamp):
    """
    保存语义分割结果的多种格式
    ========================
    
    Args:
        original_img: 原始PIL图像
        segmentation_result: 分割结果张量 (H, W)
        output_dir: 输出目录
        timestamp: 时间戳字符串
    """
    print(f"\n>>> 保存分割结果到: {output_dir}")
    
    # 1. 保存原始图像
    original_path = os.path.join(output_dir, f"original_{timestamp}.jpg")
    original_img.save(original_path, quality=95)
    print(f"  ✓ 原始图像: {original_path}")
    
    # 2. 保存分割掩码（灰度图）
    mask_array = segmentation_result.numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask_array, mode='L')
    mask_path = os.path.join(output_dir, f"mask_{timestamp}.png")
    mask_img.save(mask_path)
    print(f"  ✓ 分割掩码: {mask_path}")
    
    # 3. 保存彩色分割图
    # 使用matplotlib的colormap生成彩色版本
    plt.figure(figsize=(10, 8))
    plt.imshow(segmentation_result, cmap=colormaps["Spectral"])
    plt.axis('off')
    colored_path = os.path.join(output_dir, f"colored_{timestamp}.png")
    plt.savefig(colored_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  ✓ 彩色分割图: {colored_path}")
    
    # 4. 保存叠加图像（原图+分割结果）
    plt.figure(figsize=(15, 5))
    
    # 子图1: 原始图像
    plt.subplot(131)
    plt.imshow(original_img)
    plt.title('原始图像', fontsize=12)
    plt.axis('off')
    
    # 子图2: 分割结果
    plt.subplot(132)
    plt.imshow(segmentation_result, cmap=colormaps["Spectral"])
    plt.title('分割结果', fontsize=12)
    plt.axis('off')
    
    # 子图3: 叠加显示
    plt.subplot(133)
    plt.imshow(original_img, alpha=0.7)
    plt.imshow(segmentation_result, cmap=colormaps["Spectral"], alpha=0.5)
    plt.title('叠加显示', fontsize=12)
    plt.axis('off')
    
    overlay_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 对比图像: {overlay_path}")
    
    # 5. 保存带颜色条的完整可视化
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(original_img)
    plt.title('原始图像', fontsize=14)
    plt.axis('off')
    
    plt.subplot(122)
    im = plt.imshow(segmentation_result, cmap=colormaps["Spectral"])
    plt.title('语义分割结果', fontsize=14)
    plt.axis('off')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('类别标签', rotation=270, labelpad=20)
    
    full_viz_path = os.path.join(output_dir, f"full_visualization_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(full_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 完整可视化: {full_viz_path}")
    
    return {
        'original': original_path,
        'mask': mask_path,
        'colored': colored_path,
        'comparison': overlay_path,
        'full_visualization': full_viz_path
    }

def main(custom_image_path=None):
    """
    主函数 - 执行语义分割推理的完整流程
    ===================================
    
    Args:
        custom_image_path (str, optional): 自定义输入图像路径
    
    流程说明：
    1. 加载预训练的语义分割模型
    2. 加载并预处理测试图像
    3. 使用模型进行推理
    4. 后处理和可视化结果
    5. 保存多种格式的结果文件
    """
    print("=" * 60)
    print("DINOv3 语义分割推理开始")
    print("=" * 60)
    
    # ===== 第1步：加载预训练模型 =====
    print("\n>>> 第1步：加载DINOv3语义分割模型")
    print("模型信息：")
    print("  - 架构: ViT-7B/16 (Vision Transformer, 70亿参数)")
    print("  - 训练数据: ADE20K数据集 (150个语义类别)")
    print("  - 用途: 室内外场景的语义分割")
    
    try:
        # 注意：需要替换下面的路径为实际的权重文件路径
        segmentor = torch.hub.load(
            REPO_DIR,  # DINOv3仓库路径
            'dinov3_vit7b16_ms',  # 模型名称：ViT-7B/16多尺度语义分割器
            source="local",  # 从本地仓库加载
            # 以下两个路径需要根据实际下载的权重文件进行修改
            weights="/vepfs-perception-b/chengdu/yr/code/large_models/dinov3_weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",      # 分割器权重路径
            backbone_weights="/vepfs-perception-b/chengdu/yr/code/large_models/dinov3_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"  # 骨干网络权重路径
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("\n请确保：")
        print("1. 已下载相应的权重文件")
        print("2. 修改代码中的权重路径")
        print("3. REPO_DIR路径正确")
        return
    
    # ===== 第2步：设置推理参数 =====
    print("\n>>> 第2步：设置推理参数")
    img_size = 896  # 输入图像尺寸（像素）
    print(f"  - 输入图像尺寸: {img_size}x{img_size}")
    print(f"  - 推理模式: 滑动窗口")
    print(f"  - 输出类别数: 150 (ADE20K数据集)")
    
    # ===== 第3步：加载和预处理图像 =====
    print("\n>>> 第3步：获取并预处理测试图像")
    
    # 获取测试图像（支持自定义路径）
    img = get_img(custom_image_path)
    
    # 创建图像预处理变换
    transform = make_transform(img_size)
    print(f"✓ 图像预处理管道创建完成")
    
    # ===== 第4步：执行语义分割推理 =====
    print("\n>>> 第4步：执行语义分割推理")
    print("正在进行推理...")
    
    # 使用无梯度模式和混合精度加速推理
    with torch.inference_mode():  # 禁用梯度计算，节省内存和加速推理
        with torch.autocast('cuda', dtype=torch.bfloat16):  # 使用bfloat16混合精度
            
            # 预处理图像并添加批次维度
            batch_img = transform(img)[None]  # 形状: [1, 3, 896, 896]
            print(f"  - 预处理后图像形状: {batch_img.shape}")
            
            # 获取模型的原始预测结果（多尺度特征）
            pred_vit7b = segmentor(batch_img)  # 原始预测结果
            print(f"  - 原始预测完成")
            
            # 使用高级推理函数获得最终的分割图
            # make_inference函数实现了滑动窗口推理，能够处理高分辨率图像
            segmentation_map_vit7b = make_inference(
                batch_img,                          # 输入图像张量
                segmentor,                          # 分割模型
                inference_mode="slide",             # 推理模式：滑动窗口
                decoder_head_type="m2f",            # 解码器类型：Mask2Former
                rescale_to=(img.size[-1], img.size[-2]),  # 缩放到原始图像尺寸
                n_output_channels=150,              # ADE20K数据集有150个类别
                crop_size=(img_size, img_size),     # 滑动窗口的裁剪尺寸
                stride=(img_size, img_size),        # 滑动步长（无重叠）
                # 输出激活函数：应用softmax获得类别概率
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            ).argmax(dim=1, keepdim=True)  # 取概率最大的类别作为最终预测
            
            print(f"  - 分割推理完成")
            print(f"  - 分割图形状: {segmentation_map_vit7b.shape}")
    
    # ===== 第5步：可视化和保存结果 =====
    print("\n>>> 第5步：可视化和保存分割结果")
    
    # 将分割结果转换为CPU张量并移除批次和通道维度
    segmentation_result = segmentation_map_vit7b[0, 0].cpu()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存多种格式的结果
    saved_files = save_segmentation_results(img, segmentation_result, OUTPUT_DIR, timestamp)
    
    # 简单显示（可选）
    plt.figure(figsize=(12, 6))
    
    # 左子图：显示原始图像
    plt.subplot(121)
    plt.imshow(img)
    plt.axis("off")
    plt.title("原始图像", fontsize=14, pad=10)
    
    # 右子图：显示分割结果
    plt.subplot(122)
    plt.imshow(segmentation_result, cmap=colormaps["Spectral"])
    plt.axis("off")
    plt.title("语义分割结果", fontsize=14, pad=10)
    
    # 添加颜色条说明
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('类别标签', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()  # 显示当前结果
    
    # ===== 第6步：分析分割结果 =====
    print("\n>>> 第6步：分析分割结果")
    
    # 统计检测到的类别
    unique_classes = torch.unique(segmentation_result)
    print(f"  - 检测到 {len(unique_classes)} 个不同的语义类别")
    print(f"  - 类别索引: {unique_classes.tolist()}")
    
    # 计算每个类别的像素数量和占比
    total_pixels = segmentation_result.numel()
    print(f"  - 图像总像素数: {total_pixels}")
    
    print("\n类别分布统计:")
    for class_id in unique_classes:
        pixel_count = (segmentation_result == class_id).sum().item()
        percentage = (pixel_count / total_pixels) * 100
        print(f"    类别 {class_id}: {pixel_count} 像素 ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("语义分割推理完成！")
    print("=" * 60)
    print(f"\n📁 所有结果文件已保存到: {OUTPUT_DIR}")
    print("📋 保存的文件列表:")
    for file_type, file_path in saved_files.items():
        print(f"  • {file_type}: {os.path.basename(file_path)}")
    print("=" * 60)

def print_usage_instructions():
    """
    打印详细的使用说明
    ================
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DINOv3 语义分割使用说明                      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📋 运行此脚本前的准备工作：
    
    1️⃣  下载模型权重文件
       • 访问: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
       • 下载 ViT-7B/16 的语义分割器权重（在ADE20K数据集上训练）
       • 下载对应的骨干网络(backbone)权重文件
    
    2️⃣  修改代码配置
       • 将第47行的 <SEGMENTOR/CHECKPOINT/URL/OR/PATH> 替换为分割器权重文件路径
       • 将第48行的 <BACKBONE/CHECKPOINT/URL/OR/PATH> 替换为骨干网络权重文件路径
       • 确保第12行的 REPO_DIR 指向正确的DINOv3仓库路径
    
    3️⃣  安装Python依赖
       pip install torch torchvision pillow matplotlib requests
       
       # 如果需要CUDA支持（推荐）
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    4️⃣  硬件要求
       • 推荐: GPU (CUDA) - 显著加速推理
       • 最低: CPU - 推理会很慢但可运行
       • 内存: ViT-7B模型需要大量显存（建议40GB+的GPU或64GB+的系统内存）
    
    ⚙️  关键参数说明：
    
    • img_size (896): 输入图像尺寸，越大精度越高但速度越慢
    • inference_mode="slide": 滑动窗口推理，适合高分辨率图像
    • decoder_head_type="m2f": Mask2Former解码器，性能优秀
    • n_output_channels=150: ADE20K数据集包含150个语义类别
    • crop_size, stride: 控制滑动窗口的大小和重叠程度
    
    🎨 ADE20K数据集类别示例：
    
    室内场景: 天花板、墙壁、地板、床、桌子、椅子、沙发、电视...
    室外场景: 天空、建筑、道路、树木、汽车、行人、标志牌...
    
    🔧 故障排除：
    
    • CUDA内存不足 → 减小img_size或使用CPU
    • 模型加载失败 → 检查权重文件路径和DINOv3仓库路径
    • 推理速度慢 → 确认GPU可用性，考虑使用较小的模型
    • 导入错误 → 确保DINOv3仓库在Python路径中
    
    📊 输出说明：
    
    • 分割图中不同颜色代表不同的语义类别
    • 使用Spectral colormap进行颜色映射
    • 颜色条显示类别标签的数值范围(0-149)
    • 控制台输出包含详细的类别分布统计信息
    
    💾 保存的文件类型：
    
    • original_时间戳.jpg: 原始输入图像
    • mask_时间戳.png: 分割掩码（灰度图，每个像素值代表类别ID）
    • colored_时间戳.png: 彩色分割图（使用颜色映射）
    • comparison_时间戳.png: 三合一对比图（原图+分割+叠加）
    • full_visualization_时间戳.png: 完整可视化（带颜色条）
    
    所有文件保存在 ./segmentation_outputs/ 目录中
    """)

if __name__ == "__main__":
    # 显示使用说明
    print_usage_instructions()
    
    # 询问用户是否继续执行
    user_input = input("\n是否继续执行语义分割推理？(y/N): ").strip().lower()
    
    if user_input in ['y', 'yes', '是']:
        # 询问是否使用自定义图像
        custom_path = input("\n输入自定义图像路径 (直接回车使用示例图像): ").strip()
        if not custom_path:
            custom_path = None
        elif not os.path.exists(custom_path):
            print(f"警告: 文件 {custom_path} 不存在，将使用示例图像")
            custom_path = None
        
        try:
            # 运行主程序
            main(custom_path)
        except KeyboardInterrupt:
            print("\n\n用户中断执行")
        except Exception as e:
            print(f"\n\n执行过程中出现错误: {e}")
            print("请检查配置和依赖是否正确安装")
    else:
        print("程序退出。请先完成准备工作后再运行此脚本。")