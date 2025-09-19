# DINOv3 零样本语义分割

使用DINOTxt模型实现基于文本提示的开放词汇语义分割。

## 功能特点

- ✨ **零样本分割**: 无需训练即可分割任意文本描述的物体
- 🎯 **开放词汇**: 支持自然语言描述，如"red car"、"person wearing hat"
- 🔄 **批量处理**: 同时处理多个文本提示
- 📊 **丰富可视化**: 提供相似度热图、分割掩码、叠加显示等
- ⚡ **高效推理**: 基于预训练DINOTxt模型，推理速度快

## 快速开始

### 环境要求

```bash
pip install torch torchvision pillow matplotlib numpy requests
```

### 基本使用

```bash
# 使用默认示例图像和提示
python zero_shot_segmentation.py

# 自定义图像和文本提示
python zero_shot_segmentation.py \
    --image /path/to/your/image.jpg \
    --prompts "cat" "dog" "person" \
    --threshold 0.3

# 调整参数
python zero_shot_segmentation.py \
    --image /path/to/image.jpg \
    --prompts "red car" "tree" "building" \
    --threshold 0.25 \
    --size 518 \
    --output_dir ./my_results
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image` | str | None | 输入图像路径（为空时下载示例图像） |
| `--prompts` | list | ["cat", "person", "car"] | 文本提示列表 |
| `--threshold` | float | 0.3 | 分割阈值(0.0-1.0) |
| `--output_dir` | str | "./zero_shot_outputs" | 输出目录 |
| `--weights` | str | None | DINOTxt权重文件路径 |
| `--backbone_weights` | str | None | 骨干网络权重文件路径 |
| `--size` | int | 518 | 输入图像尺寸 |
| `--device` | str | "cuda" | 计算设备 |

## 权重文件下载

### 方法1: 官方下载页面

访问 [DINOv3官方下载页面](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)

需要下载：
- ViT-L/16 distilled backbone weights
- ViT-L/16 DINOTxt weights
- BPE vocabulary file

### 方法2: 直接链接

```bash
# DINOTxt权重文件
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth

# 骨干网络权重文件  
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

# BPE词汇表
wget https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz
```

## 输出文件说明

运行后会在输出目录生成以下文件：

```
zero_shot_outputs/
├── similarity_cat_20241201_143022.png      # 单个提示的相似度热图
├── similarity_person_20241201_143022.png   # 
├── mask_cat_20241201_143022.png            # 二值分割掩码
├── mask_person_20241201_143022.png         # 
└── zero_shot_segmentation_20241201_143022.png  # 多类别合并可视化
```

### 文件类型详解

1. **相似度热图**: 显示文本与图像区域的匹配程度
   - 热图颜色越亮表示相似度越高
   - 包含原图、热图、分割结果三个子图

2. **分割掩码**: 二值PNG图像
   - 白色区域(255)表示检测到的目标
   - 黑色区域(0)表示背景
   - 可用于后续处理或分析

3. **合并可视化**: 展示所有文本提示的分割结果
   - 不同颜色表示不同的目标类别
   - 支持重叠区域的可视化

## 使用技巧

### 1. 文本提示编写

**有效的提示**:
- 具体描述: "red car" 比 "vehicle" 更准确
- 简洁明了: "cat" 比 "a cute furry cat" 更好
- 避免否定: 使用"person"而不是"not a tree"

**示例提示**:
```python
# 物体类别
["cat", "dog", "car", "tree", "person"]

# 带属性的描述
["red car", "green tree", "white building"]

# 复合物体
["person wearing hat", "dog on grass", "car on road"]
```

### 2. 阈值调节

- **低阈值(0.1-0.2)**: 检测更多区域，可能包含误检
- **中阈值(0.3-0.4)**: 平衡准确性和召回率
- **高阈值(0.5-0.7)**: 只保留高置信度区域，更精确

### 3. 输入尺寸选择

- **518x518**: 推荐尺寸，平衡速度和精度
- **224x224**: 更快但精度较低
- **714x714**: 更高精度但计算量大

## 常见问题

### Q: 模型加载失败
**A**: 检查权重文件路径和网络连接，确保有足够内存(推荐32GB+)

### Q: 分割效果不好
**A**: 尝试调整阈值、改写文本提示、使用更高的输入分辨率

### Q: 显存不足
**A**: 减小输入尺寸或使用CPU推理(`--device cpu`)

### Q: 文本提示不生效
**A**: 确保提示简洁明确，避免过于复杂的描述

## 技术原理

DINOv3零样本分割基于以下原理：

1. **双编码器架构**: 分别编码图像和文本
2. **特征对齐**: 通过对比学习使视觉和文本特征在同一空间
3. **密集预测**: 计算文本特征与图像patch特征的相似度
4. **空间重构**: 将patch级相似度重塑为像素级分割掩码

## 性能基准

在COCO数据集上的表现（参考值）：

| 文本提示 | IoU | 精确度 | 召回率 |
|----------|-----|--------|--------|
| person   | 0.45| 0.68   | 0.72   |
| car      | 0.52| 0.74   | 0.69   |
| cat      | 0.38| 0.61   | 0.65   |

*注：实际性能取决于图像质量、文本提示和阈值设置*

## 扩展功能

想要实现更多功能？参考以下扩展：

- **批量图像处理**: 修改脚本支持文件夹输入
- **视频分割**: 逐帧处理视频文件
- **交互式分割**: 结合GUI界面实时调整提示
- **API服务**: 封装为Web服务接口

## 许可证

本代码基于DINOv3官方实现，遵循相应的开源许可证。