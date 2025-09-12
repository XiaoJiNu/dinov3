# ADE20K语义分割训练指南 - 基于DINOv3

本实现提供了基于DINOv3模型在ADE20K数据集上进行语义分割的完整训练和推理脚本。

## 🚀 快速开始

### 环境要求
- PyTorch 2.1+ (使用pytorch21环境: `/home/yr/anaconda3/envs/pytorch21`)
- CUDA支持 (已验证RTX 4070)
- ADE20K数据集在 `/home/yr/yr/data/ADE20K_2016_07_26`

### 立即开始训练

```bash
# 激活环境
source /home/yr/anaconda3/bin/activate pytorch21

# 快速测试 (10个样本)
python ade20k_train.py --epochs 2 --batch_size 2 --subset_size 10

# 完整训练
python ade20k_train.py --epochs 10 --batch_size 4
```

### 立即开始推理

训练完成后，使用训练好的模型进行推理：

```bash
# 对单张图像进行推理
python ade20k_inference.py --input /path/to/your/image.jpg

# 批量推理整个目录
python ade20k_inference.py --input /path/to/images/ --output_dir ./my_results/
```

## 📚 主要特性

- **✅ 完整ADE20K支持**: 
  - 自动处理层级目录结构 (20,210个训练样本)
  - 正确解析RG编码的分割标签
  - 150个标准ADE20K类别

- **✅ 多种训练方法**: 
  - Linear segmentation head (推荐)
  - Logistic regression  
  - MLP classifier

- **✅ 本地模型支持**: 
  - 优先使用本地DINOv3模型
  - 自动回退到在线下载

## 📁 数据集结构

ADE20K数据集实际结构 (已自动识别20,210个训练样本):
```
/home/yr/yr/data/ADE20K_2016_07_26/
└── images/
    ├── training/           # 训练集
    │   ├── a/             # 按首字母分组
    │   │   ├── abbey/     # 场景类别目录
    │   │   │   ├── *.jpg          # RGB图像
    │   │   │   └── *_seg.png      # 分割标签
    │   │   └── airport_terminal/
    │   ├── b/
    │   │   ├── bar/
    │   │   └── bedroom/
    │   └── ...
    └── validation/         # 验证集 (相同结构)
        └── ...
```

### 🔧 数据格式说明:
- **RGB图像**: `*.jpg` 文件包含输入图像
- **分割标签**: `*_seg.png` 文件包含语义分割标签
  - **R和G通道**: 编码对象类别 (类别ID = R + G × 256)
  - **B通道**: 编码实例ID
  - **索引转换**: 原始1基索引自动转换为0基索引
- **类别数量**: 150个ADE20K标准类别

## 🎯 使用方法

### 训练脚本

**主推荐脚本: `ade20k_train.py`** (经过优化，支持本地模型):
```bash
# 激活环境
source /home/yr/anaconda3/bin/activate pytorch21

# 快速测试 (小数据集)
python ade20k_train.py --epochs 2 --batch_size 2 --subset_size 50

# 标准训练
python ade20k_train.py --epochs 10 --batch_size 4

# 完整训练 (长时间)
python ade20k_train.py --epochs 20 --batch_size 8
```

**备用脚本: `semantic_train.py`** (通用版本):
```bash
# Linear头训练 (推荐)
python semantic_train.py --ade20k_root /home/yr/yr/data/ADE20K_2016_07_26 --method linear --epochs 20 --batch_size 4

# 逻辑回归 (基于patch)
python semantic_train.py --ade20k_root /home/yr/yr/data/ADE20K_2016_07_26 --method logistic

# MLP分类器
python semantic_train.py --ade20k_root /home/yr/yr/data/ADE20K_2016_07_26 --method mlp
```

### 推理

训练完成后，有两种推理脚本可选：

#### 方法1: ade20k_inference.py (推荐)

专门针对 ADE20K 优化的推理脚本，支持多种可视化输出：

```bash
# 单张图像推理
python ade20k_inference.py --input /path/to/image.jpg

# 批量图像推理
python ade20k_inference.py --input /path/to/images/ --output_dir ./results/

# 自定义参数
python ade20k_inference.py \
    --input ./image.jpg \
    --model_path ./ade20k_results/best_ade20k_seg_head.pth \
    --dinov3_path ./models/dinov3.pth \
    --image_size 384 \
    --output_dir ./custom_output/
```

**ade20k_inference.py 参数说明：**
- `--input`: 输入图像路径或目录 (必需)
- `--model_path`: 训练好的分割头模型路径 (默认: `./ade20k_results/best_ade20k_seg_head.pth`)
- `--output_dir`: 输出目录 (默认: `./ade20k_outputs`)
- `--image_size`: 输入图像尺寸 (默认: 512)
- `--dinov3_path`: DINOv3 模型路径 (默认: 本地模型路径)

**支持的图像格式：** JPEG (.jpg, .jpeg), PNG (.png), BMP (.bmp)

**输出文件 (每张图像生成4个文件)：**
- `{image_name}_mask.png`: 原始分割掩码 (灰度图)
- `{image_name}_colored.png`: 彩色分割结果
- `{image_name}_overlay.png`: 分割结果叠加到原图
- `{image_name}_visualization.png`: 四宫格完整可视化

#### 方法2: semantic_inference.py (通用版本)

```bash
# 单张图像推理
python semantic_inference.py --model_dir ./ade20k_results --input /path/to/image.jpg --visualize

# 批量图像推理
python semantic_inference.py --model_dir ./ade20k_results --input /path/to/images/ --visualize
```

## 🛠 命令行参数

### ade20k_train.py 参数:
- `--epochs`: 训练轮数 (默认: 10)
- `--lr`: 学习率 (默认: 1e-3)
- `--batch_size`: 批次大小 (默认: 4)
- `--subset_size`: 使用子集大小，用于测试 (可选)
- `--ade20k_root`: ADE20K数据集根目录 (默认: /home/yr/yr/data/ADE20K_2016_07_26)
- `--model_path`: DINOv3模型路径 (默认: 本地路径)
- `--output_dir`: 输出目录 (默认: ./ade20k_results)

### semantic_train.py 参数:
- `--method`: 训练方法 - linear, logistic, mlp (默认: linear)
- `--epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批次大小 (默认: 4)
- `--lr`: 学习率 (默认: 1e-3)
- `--visualize`: 创建验证图表

### semantic_inference.py 参数:
- `--dinov3_path`: DINOv3 模型路径
- `--model_dir`: 包含训练模型和元数据的目录
- `--input`: 输入图像路径或目录
- `--output_dir`: 输出目录 (默认: ./ade20k_inference_outputs)
- `--visualize`: 创建可视化图表
- `--device`: 推理设备 (cuda/cpu)

## 📄 输出文件

### 训练输出 (ade20k_results/)
- `best_ade20k_seg_head.pth`: 最佳模型权重
- `final_ade20k_seg_head.pth`: 最终模型权重
- `training_history.json`: 训练历史记录
- `class_names.json`: ADE20K类别名称 (如果使用semantic_train.py)
- `metadata.json`: 模型配置参数 (如果使用semantic_train.py)

### 推理输出

#### ade20k_inference.py 输出 (ade20k_outputs/)
每张输入图像生成以下4个文件：
- `{image_name}_mask.png`: 原始分割掩码 (灰度图像，像素值对应类别ID)
- `{image_name}_colored.png`: 彩色分割结果 (使用ADE20K颜色映射)
- `{image_name}_overlay.png`: 分割结果叠加到原始图像
- `{image_name}_visualization.png`: 四宫格完整可视化 (原图+掩码+彩色+叠加)

#### semantic_inference.py 输出 (ade20k_inference_outputs/)
- `{image_name}_prediction.png`: 原始分割掩码
- `{image_name}_prediction_filtered.png`: 平滑后分割掩码
- `{image_name}_prediction_colored.png`: 彩色分割结果
- `{image_name}_visualization.png`: 完整可视化 (使用--visualize时)
- `{image_name}_probabilities/`: 每类别概率图目录
- `{image_name}_statistics.json`: 类别分布统计

## 📋 ADE20K类别 (150类)

包括常见场景对象:
wall, building, sky, floor, tree, ceiling, road, bed, windowpane, grass, cabinet, sidewalk, person, earth, door, table, mountain, plant, curtain, chair, car, water, painting, sofa, shelf, house, sea, mirror, rug, field, armchair, seat, fence, desk, rock, wardrobe, lamp, bathtub, railing, cushion, 等等...

完整的150个类别列表会自动保存在训练输出的`class_names.json`文件中。

## 💡 性能优化建议

1. **推荐配置**: 
   - **ade20k_train.py** + Linear头: 最佳精度，需要更多GPU内存
   - batch_size=4-8 (取决于GPU内存)
   - epochs=10-20 (根据收敛情况调整)

2. **内存管理**:
   - GPU内存不足时减少batch_size
   - 使用--subset_size进行快速测试
   - Linear方法需要最多内存，但精度最高

3. **数据处理**: 
   - 自动处理ADE20K的RG通道编码
   - 自动发现层级目录结构
   - 支持20,210个训练样本的完整加载

4. **模型选择**:
   - **Linear头**: 最佳选择，端到端训练
   - **Logistic回归**: 适合快速实验
   - **MLP**: 平衡精度和速度

## Requirements

- PyTorch
- transformers
- PIL (Pillow)
- numpy
- matplotlib
- scikit-learn
- scipy
- tqdm

## ⚠️ 重要说明

### 模型路径问题
当前本地模型路径可能需要调整。如遇到模型加载错误，脚本会自动回退到在线下载。

### 数据格式细节

ADE20K数据集使用特殊的编码格式:
- **对象类别**: 在R和G通道中编码，公式为 `类别ID = R + G × 256`
- **实例ID**: 在B通道中编码
- **索引转换**: 原始ADE20K使用1基索引，自动转换为0基索引
- **背景**: 类别0代表背景/未标记像素

## 🔧 故障排除

### 训练相关
1. **模型加载失败**: 检查网络连接，脚本会自动尝试在线下载
2. **内存不足**: 减少batch_size或使用subset_size
3. **CUDA错误**: 确保pytorch21环境正确安装CUDA支持
4. **数据集路径**: 确认数据集在 `/home/yr/yr/data/ADE20K_2016_07_26`

### 推理相关 (ade20k_inference.py)
1. **模型文件不存在错误**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '...'
   ```
   **解决方案**: 检查并更新模型文件路径，确保训练好的分割头模型存在

2. **CUDA 内存不足**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案**: 减小 `--image_size` 参数 (如 `--image_size 384`) 或强制使用 CPU

3. **DINOv3 模型加载失败**:
   ```
   RuntimeError: 所有模型加载方法都失败
   ```
   **解决方案**: 
   - 检查 torch.hub 缓存: `torch.hub.list('facebookresearch/dinov3')`
   - 确认网络连接用于在线下载
   - 检查本地模型路径设置

4. **不支持的图像格式**:
   ```
   不支持的文件格式: .xxx
   ```
   **解决方案**: 仅支持 .jpg, .jpeg, .png, .bmp 格式

5. **图像预处理错误**:
   ```
   OSError: cannot identify image file
   ```
   **解决方案**: 检查图像文件是否损坏，确认文件完整性

## 📞 支持

- 训练脚本已验证可以正确加载20,210个训练样本
- 支持150个ADE20K标准类别
- 经过PyTorch 2.5.1和CUDA环境测试
- 所有脚本都通过语法检查和基本功能测试