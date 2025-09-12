#!/usr/bin/env python3
"""
ADE20K训练脚本 - 基于example/segment_train.py，使用transformers库
用ADE20K数据集训练语义分割模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoImageProcessor
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import glob
from torchvision import transforms

# 导入dinov3官方数据加载器
from dinov3.data.datasets.ade20k import ADE20K, _Split

# 模型配置
REPO_DIR = "/home/yr/yr/code/cv/large_models/dinov3_all/dinov3"
WEIGHTS_PATH = "/home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
LOCAL_MODEL_PATH = "/home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"  
N_CLASSES = 150  # ADE20K有150个类别


class LinearSegHead(nn.Module):  
    """线性分割头"""
    def __init__(self, in_ch, n_classes):  
        super().__init__()  
        self.proj = nn.Conv2d(in_ch, n_classes, 1)  
        self.up = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)  
    
    def forward(self, fmap): 
        return self.up(self.proj(fmap))  


def extract_fmap(model, proc, image):  
    """从DINOv3模型提取特征图"""
    inputs = proc(images=image, return_tensors="pt").to(model.device)  
    with torch.inference_mode():  
        out = model(**inputs)  
    num_regs = model.config.num_register_tokens  
    grid = out.last_hidden_state[:, 1 + num_regs:, :]  # 丢弃CLS+寄存器
    B, N, C = grid.shape  
    H = W = int(N ** 0.5)  
    return grid.reshape(B, H, W, C).permute(0, 3, 1, 2)  



class ADE20KWrapper(Dataset):
    """包装dinov3官方ADE20K数据加载器的类"""
    
    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 224):
        self.image_size = image_size
        
        # 使用dinov3官方数据加载器
        dinov3_split = _Split.TRAIN if split == 'train' else _Split.VAL
        self.dataset = ADE20K(split=dinov3_split, root=root_dir)
        
        # 图像预处理transform
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 标签预处理transform
        self.target_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])
        
        print(f"Loaded {len(self.dataset)} samples from {split} split (image_size={self.image_size})")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 使用dinov3数据加载器获取原始数据
        image, target = self.dataset[idx]
        
        # 处理图像
        image_tensor = self.image_transform(image)
        
        # 处理标签
        target_array = np.array(target)
        if target_array.ndim == 3:
            target_array = target_array[:, :, 0]
        
        # 正确的类别映射：0(背景) -> 255(ignore), 1-150 -> 0-149
        target_array = target_array.astype(np.int64)
        target_array[target_array == 0] = 255  # 背景设为ignore_index
        valid_mask = target_array != 255
        target_array[valid_mask] = target_array[valid_mask] - 1  # 1-150 -> 0-149
        
        # 转换为tensor并resize
        target_pil = Image.fromarray(target_array.astype(np.uint8))
        target_resized = self.target_transform(target_pil)
        target_final = torch.from_numpy(np.array(target_resized)).long()
        
        return image_tensor, target_final


def compute_miou(pred, target, num_classes, ignore_index=255):
    """计算平均IoU"""
    ious = []
    for cls in range(num_classes):
        # 忽略指定的索引
        valid_mask = target != ignore_index
        pred_cls = (pred == cls) & valid_mask
        target_cls = (target == cls) & valid_mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    # 计算平均IoU，忽略nan值
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0, ious

def compute_streaming_miou(intersection_counts, union_counts, num_classes):
    """从累积的交集和并集计数计算mIoU"""
    ious = []
    for cls in range(num_classes):
        if union_counts[cls] > 0:
            iou = intersection_counts[cls] / union_counts[cls]
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0, ious


def resize_mask_to_logits(mask, logits_shape):
    """将mask调整为与logits相同的大小"""
    from torchvision.transforms import functional as TF
    
    # mask: (H, W), logits_shape: (B, C, H', W')
    target_h, target_w = logits_shape[2], logits_shape[3]
    
    # 添加batch和channel维度进行resize
    mask_resized = TF.resize(
        mask.float().unsqueeze(0).unsqueeze(0),
        (target_h, target_w),
        interpolation=TF.InterpolationMode.NEAREST
    ).squeeze().long()
    
    return mask_resized


def load_dinov3_model(model_path=LOCAL_MODEL_PATH):
    """加载本地DINOv3模型"""
    try:
        # 方法1: 使用正确的torch.hub.load方式加载本地模型
        print(f"尝试从DINOv3 repo加载模型: {REPO_DIR}")
        print(f"使用权重文件: {WEIGHTS_PATH}")
        model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH, trust_repo=True)
        # model = torch.hub.load(weights=WEIGHTS_PATH)
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
    """从本地DINOv3模型提取特征图"""
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




def save_checkpoint(epoch, head, optimizer, train_losses, val_mious, best_miou, output_dir):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_mious': val_mious,
        'best_miou': best_miou
    }
    checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最新检查点的链接
    latest_checkpoint_path = f"{output_dir}/latest_checkpoint.pth"
    torch.save(checkpoint, latest_checkpoint_path)
    print(f"最新检查点已保存: {latest_checkpoint_path}")


def load_checkpoint(checkpoint_path, head, optimizer, device):
    """加载训练检查点"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    head.load_state_dict(checkpoint['head_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    train_losses = checkpoint.get('train_losses', [])
    val_mious = checkpoint.get('val_mious', [])
    best_miou = checkpoint.get('best_miou', 0.0)
    
    print(f"检查点加载完成，从epoch {start_epoch} 开始训练")
    print(f"最佳mIoU: {best_miou:.4f}")
    
    return start_epoch, train_losses, val_mious, best_miou


def main(epochs=10, lr=1e-3, batch_size=4, subset_size=None, 
         ade20k_root="/home/yr/yr/data/ade20k/ade/ADEChallengeData2016", 
         output_dir="./ade20k_results",
         model_path=LOCAL_MODEL_PATH,
         image_size=224,
         resume=None):
    """主训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("ADE20K语义分割训练")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print(f"数据集路径: {ade20k_root}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {image_size}x{image_size}")
    if subset_size:
        print(f"使用子集大小: {subset_size}")
    if resume:
        print(f"恢复训练: {resume}")
    print("=" * 50)
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 加载模型
    print("加载DINOv3模型...")
    model, proc = load_dinov3_model(model_path)
    
    # 将模型移动到GPU
    model = model.to(device)
    
    # 冻结DINOv3参数
    for p in model.parameters():
        p.requires_grad_(False)
    
    # 创建数据集
    print("创建数据集...")
    train_set = ADE20KWrapper(ade20k_root, 'train', image_size)
    val_set = ADE20KWrapper(ade20k_root, 'val', image_size)
    
    # 如果指定了子集大小，则使用子集
    if subset_size and subset_size < len(train_set):
        from torch.utils.data import Subset
        indices = torch.randperm(len(train_set))[:subset_size]
        train_set = Subset(train_set, indices)
        print(f"使用训练子集: {len(train_set)} 样本")
    
    # 推断通道数
    print("推断特征维度...")
    x0, _ = train_set[0]
    if proc is not None:
        fmap0 = extract_fmap(model, proc, x0)
    else:
        fmap0 = extract_fmap_local(model, x0)
    feature_dim = fmap0.shape[1]
    print(f"特征维度: {feature_dim}")
    
    # 创建分割头
    head = LinearSegHead(feature_dim, N_CLASSES).to(device)
    opt = optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2) # batch_size
    
    print(f"训练样本: {len(train_set)}")
    print(f"验证样本: {len(val_set)}")
    print()
    
    # 初始化训练状态
    start_epoch = 0
    best_miou = 0.0
    train_losses = []
    val_mious = []
    
    # 如果指定了恢复训练的检查点
    if resume:
        if resume == 'latest':
            resume_path = f"{output_dir}/latest_checkpoint.pth"
        else:
            resume_path = resume
            
        if Path(resume_path).exists():
            start_epoch, train_losses, val_mious, best_miou = load_checkpoint(
                resume_path, head, opt, device
            )
        else:
            print(f"警告: 检查点文件不存在: {resume_path}")
            print("从头开始训练...")
    
    for ep in range(start_epoch, epochs):
        print(f"Epoch {ep+1}/{epochs}")
        print("-" * 30)
        
        # 训练阶段
        head.train()
        epoch_losses = []
        
        for batch_idx, (img_batch, mask_batch) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
            # 提取特征
            fmaps = []
            for img_tensor in img_batch:
                if proc is not None:
                    # 使用transformers模型
                    fmap = extract_fmap(model, proc, img_tensor)
                else:
                    # 使用本地模型
                    fmap = extract_fmap_local(model, img_tensor)
                fmaps.append(fmap)
            fmaps = torch.cat(fmaps, dim=0)
            
            # 前向传播
            logits = head(fmaps)
            
            # 将mask移动到GPU并调整大小以匹配logits
            mask_batch = mask_batch.to(device)
            masks_resized = []
            for mask in mask_batch:
                mask_resized = resize_mask_to_logits(mask, logits.shape)
                masks_resized.append(mask_resized)
            masks_resized = torch.stack(masks_resized).to(device)
            
            # 计算损失
            loss = loss_fn(logits, masks_resized)
            
            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 500 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}", end='\r', flush=True)
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        print(f"平均训练损失: {avg_train_loss:.4f}")
        
        # 保存当前epoch模型和检查点
        torch.save(head.state_dict(), f"{output_dir}/epoch_{ep}.pth")
        save_checkpoint(ep, head, opt, train_losses, val_mious, best_miou, output_dir)
        
        # 清理旧的检查点，只保留最近5个
        cleanup_old_checkpoints(output_dir, keep_last=5)

        # 验证阶段 - 使用流式计算避免内存爆炸
        if ep % 2 == 0:  # 每2个epoch验证一次
            print("验证中...")
            head.eval()
            
            # 初始化IoU统计计数器
            intersection_counts = torch.zeros(N_CLASSES, dtype=torch.float64)
            union_counts = torch.zeros(N_CLASSES, dtype=torch.float64)
            total_pixels = 0
            max_val_batches = 200  # 限制验证batch数量以节省时间和内存
            
            with torch.no_grad():
                for batch_idx, (img_batch, mask_batch) in enumerate(tqdm(val_loader, desc="Validating", ncols=100)):
                    if batch_idx >= max_val_batches:
                        break
                    
                    # 提取特征
                    fmaps = []
                    for img_tensor in img_batch:
                        if proc is not None:
                            # 使用transformers模型
                            fmap = extract_fmap(model, proc, img_tensor)
                        else:
                            # 使用本地模型
                            fmap = extract_fmap_local(model, img_tensor)
                        fmaps.append(fmap)
                    fmaps = torch.cat(fmaps, dim=0)
                    
                    # 前向传播
                    logits = head(fmaps)
                    preds = torch.argmax(logits, dim=1)
                    
                    # 将mask移动到GPU
                    mask_batch = mask_batch.to(device)
                    
                    # 流式计算IoU统计
                    for class_id in range(N_CLASSES):
                        # 创建类别掩码
                        pred_mask = (preds == class_id)
                        target_mask = (mask_batch == class_id)
                        
                        # 忽略背景像素 (255)
                        valid_mask = mask_batch != 255
                        pred_mask = pred_mask & valid_mask
                        target_mask = target_mask & valid_mask
                        
                        # 计算交集和并集
                        intersection = (pred_mask & target_mask).sum().float().cpu()
                        union = (pred_mask | target_mask).sum().float().cpu()
                        
                        # 累积统计
                        intersection_counts[class_id] += intersection
                        union_counts[class_id] += union
                    
                    # 统计总像素数
                    total_pixels += (mask_batch != 255).sum().item()
                    
                    # 立即清理GPU内存
                    del logits, preds, fmaps
                    torch.cuda.empty_cache()
            
            # 计算最终mIoU
            miou, _ = compute_streaming_miou(intersection_counts, union_counts, N_CLASSES)
            val_mious.append(miou)
            
            print(f"验证 mIoU: {miou:.4f} (基于 {batch_idx+1}/{len(val_loader)} batches, {total_pixels} pixels)")
            
            # 保存最佳模型
            if miou > best_miou:
                best_miou = miou
                torch.save(head.state_dict(), f"{output_dir}/best_ade20k_seg_head.pth")
                print(f"保存最佳模型 (mIoU: {best_miou:.4f})")
        
        print()
    
    # 保存最终模型
    torch.save(head.state_dict(), f"{output_dir}/final_ade20k_seg_head.pth")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_mious': val_mious,
        'best_miou': best_miou,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr
    }
    
    with open(f"{output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("训练完成！")
    print(f"最佳 mIoU: {best_miou:.4f}")
    print(f"结果保存到: {output_dir}")
    
    # 保存最终检查点
    save_checkpoint(epochs-1, head, opt, train_losses, val_mious, best_miou, output_dir)


def cleanup_old_checkpoints(output_dir, keep_last=5):
    """清理旧的检查点文件，只保留最新的几个"""
    checkpoint_pattern = f"{output_dir}/checkpoint_epoch_*.pth"
    checkpoints = glob.glob(checkpoint_pattern)
    
    # 按文件修改时间排序
    checkpoints.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    # 删除超出保留数量的检查点
    for checkpoint in checkpoints[keep_last:]:
        try:
            Path(checkpoint).unlink()
            print(f"清理旧检查点: {checkpoint}")
        except Exception as e:
            print(f"清理检查点失败 {checkpoint}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADE20K语义分割训练")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=6, help="批次大小")
    parser.add_argument("--subset_size", type=int, default=None, help="使用数据子集大小（用于快速测试）") #
    parser.add_argument("--ade20k_root", type=str, 
                       default="/home/yr/yr/data/ade20k/ade/ADEChallengeData2016",
                       help="ADE20K数据集根目录")
    parser.add_argument("--model_path", type=str, 
                       default=LOCAL_MODEL_PATH,
                       help="DINOv3模型路径")
    parser.add_argument("--output_dir", type=str, default="./ade20k_results",
                       help="输出目录")
    parser.add_argument("--image_size", type=int, default=448,
                       help="输入图像尺寸")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的检查点路径，或者使用'latest'加载最新检查点")
    
    args = parser.parse_args()
    
    main(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        ade20k_root=args.ade20k_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        resume=args.resume
    )