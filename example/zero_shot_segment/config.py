#!/usr/bin/env python3
"""
DINOv3 零样本分割配置文件
========================
集中管理模型路径、参数设置和预设配置
"""

import os
from typing import Dict, List, Tuple

# ===== 基础配置 =====
class BaseConfig:
    """基础配置类"""
    
    # 路径配置
    REPO_DIR = "/vepfs-perception-b/chengdu/yr/code/large_models/dinov3"
    TORCH_CACHE_DIR = "/vepfs-perception-b/chengdu/yr/cache/torch"
    WEIGHTS_DIR = "/vepfs-perception-b/chengdu/yr/code/large_models/dinov3_weights"
    
    # 默认权重文件
    DEFAULT_DINOTXT_WEIGHTS = os.path.join(WEIGHTS_DIR, "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth")
    DEFAULT_BACKBONE_WEIGHTS = os.path.join(WEIGHTS_DIR, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    DEFAULT_BPE_VOCAB = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
    
    # 模型配置
    DEFAULT_IMAGE_SIZE = 518  # 518 = 14 * 37，适合ViT-L/16
    DEFAULT_PATCH_SIZE = 14
    DEFAULT_EMBED_DIM = 2048
    
    # 推理配置
    DEFAULT_THRESHOLD = 0.3
    DEFAULT_DEVICE = "cuda"
    DEFAULT_OUTPUT_DIR = "./zero_shot_outputs"

# ===== 预设配置 =====
class PresetConfigs:
    """预设配置，针对不同使用场景"""
    
    # 快速推理配置（速度优先）
    FAST_CONFIG = {
        'image_size': 224,
        'threshold': 0.4,
        'device': 'cuda',
        'prompts': ["person", "car", "animal"]
    }
    
    # 精确推理配置（精度优先）
    ACCURATE_CONFIG = {
        'image_size': 714,  # 714 = 14 * 51
        'threshold': 0.25,
        'device': 'cuda',
        'prompts': ["person", "car", "tree", "building", "animal"]
    }
    
    # CPU推理配置（兼容性优先）
    CPU_CONFIG = {
        'image_size': 224,
        'threshold': 0.35,
        'device': 'cpu',
        'prompts': ["person", "car"]
    }
    
    # 室内场景配置
    INDOOR_CONFIG = {
        'image_size': 518,
        'threshold': 0.3,
        'prompts': [
            "person", "chair", "table", "sofa", "bed", "tv", "laptop", 
            "book", "clock", "vase", "scissors", "teddy bear", "bottle"
        ]
    }
    
    # 室外场景配置
    OUTDOOR_CONFIG = {
        'image_size': 518,
        'threshold': 0.3,
        'prompts': [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "tree", "traffic light", "stop sign", "bench", "bird",
            "cat", "dog", "horse", "cow", "elephant", "bear", "zebra"
        ]
    }
    
    # 交通场景配置
    TRAFFIC_CONFIG = {
        'image_size': 518,
        'threshold': 0.35,
        'prompts': [
            "car", "truck", "bus", "motorcycle", "bicycle", "person",
            "traffic light", "stop sign", "parking meter", "bench"
        ]
    }
    
    # 动物检测配置
    ANIMAL_CONFIG = {
        'image_size': 518,
        'threshold': 0.3,
        'prompts': [
            "cat", "dog", "bird", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag"
        ]
    }

# ===== 模型配置 =====
class ModelConfig:
    """模型相关配置"""
    
    # 支持的图像尺寸（必须是14的倍数）
    SUPPORTED_SIZES = [224, 294, 364, 434, 504, 518, 574, 644, 714]
    
    # 颜色配置（用于可视化）
    VISUALIZATION_COLORS = [
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
        [255, 20, 147],   # 深粉红
        [0, 191, 255],    # 深天蓝
        [255, 69, 0],     # 红橙色
        [50, 205, 50],    # 绿黄色
        [138, 43, 226],   # 紫罗兰
    ]
    
    # ImageNet标准化参数
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# ===== 常用文本提示 =====
class TextPrompts:
    """预定义的文本提示集合"""
    
    # COCO类别（80个）
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    # 日常物品
    DAILY_OBJECTS = [
        "person", "face", "hand", "car", "tree", "building", "sky", "road",
        "grass", "flower", "water", "mountain", "cloud", "sun", "moon"
    ]
    
    # 室内物品
    INDOOR_OBJECTS = [
        "chair", "table", "sofa", "bed", "tv", "laptop", "book", "cup",
        "bottle", "phone", "clock", "lamp", "window", "door", "wall"
    ]
    
    # 室外物品
    OUTDOOR_OBJECTS = [
        "car", "tree", "building", "road", "sky", "cloud", "grass", "flower",
        "mountain", "water", "bridge", "traffic light", "sign", "bench"
    ]
    
    # 动物类别
    ANIMALS = [
        "cat", "dog", "bird", "horse", "cow", "sheep", "elephant", "lion",
        "tiger", "bear", "zebra", "giraffe", "monkey", "rabbit", "mouse"
    ]
    
    # 交通工具
    VEHICLES = [
        "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat",
        "train", "helicopter", "taxi", "ambulance", "fire truck"
    ]

# ===== 工具函数 =====
def get_config(preset_name: str = None) -> Dict:
    """
    获取指定的预设配置
    
    Args:
        preset_name: 预设配置名称
        
    Returns:
        配置字典
    """
    preset_map = {
        'fast': PresetConfigs.FAST_CONFIG,
        'accurate': PresetConfigs.ACCURATE_CONFIG,
        'cpu': PresetConfigs.CPU_CONFIG,
        'indoor': PresetConfigs.INDOOR_CONFIG,
        'outdoor': PresetConfigs.OUTDOOR_CONFIG,
        'traffic': PresetConfigs.TRAFFIC_CONFIG,
        'animal': PresetConfigs.ANIMAL_CONFIG,
    }
    
    if preset_name is None:
        return {
            'image_size': BaseConfig.DEFAULT_IMAGE_SIZE,
            'threshold': BaseConfig.DEFAULT_THRESHOLD,
            'device': BaseConfig.DEFAULT_DEVICE,
            'prompts': TextPrompts.DAILY_OBJECTS[:5]
        }
    
    return preset_map.get(preset_name.lower(), preset_map['fast'])

def validate_image_size(size: int) -> int:
    """
    验证并调整图像尺寸
    
    Args:
        size: 输入尺寸
        
    Returns:
        有效的尺寸
    """
    if size in ModelConfig.SUPPORTED_SIZES:
        return size
    
    # 找到最接近的有效尺寸
    closest_size = min(ModelConfig.SUPPORTED_SIZES, key=lambda x: abs(x - size))
    print(f"警告: 尺寸 {size} 不是14的倍数，自动调整为 {closest_size}")
    return closest_size

def get_text_prompts(category: str) -> List[str]:
    """
    获取指定类别的文本提示
    
    Args:
        category: 类别名称
        
    Returns:
        文本提示列表
    """
    category_map = {
        'coco': TextPrompts.COCO_CLASSES,
        'daily': TextPrompts.DAILY_OBJECTS,
        'indoor': TextPrompts.INDOOR_OBJECTS,
        'outdoor': TextPrompts.OUTDOOR_OBJECTS,
        'animals': TextPrompts.ANIMALS,
        'vehicles': TextPrompts.VEHICLES,
    }
    
    return category_map.get(category.lower(), TextPrompts.DAILY_OBJECTS)

# ===== 示例使用 =====
if __name__ == "__main__":
    # 示例：获取不同配置
    print("快速配置:", get_config('fast'))
    print("精确配置:", get_config('accurate'))
    print("室内配置:", get_config('indoor'))
    
    # 示例：获取文本提示
    print("COCO类别数量:", len(get_text_prompts('coco')))
    print("动物类别:", get_text_prompts('animals')[:10])
    
    # 示例：验证图像尺寸
    print("验证尺寸 500:", validate_image_size(500))
    print("验证尺寸 518:", validate_image_size(518))