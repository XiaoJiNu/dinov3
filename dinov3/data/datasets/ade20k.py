# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
ADE20K数据集加载器

ADE20K数据集是一个大规模的场景解析数据集，包含：
- 训练集：20,210张图像
- 验证集：2,000张图像 
- 150个语义类别
- 密集标注的像素级分割标签

数据集结构：
- images/training/: 训练图像
- images/validation/: 验证图像
- annotations/: 分割标注文件
- ADE20K_object150_train.txt: 训练集文件列表
- ADE20K_object150_val.txt: 验证集文件列表
"""

import os
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

# 导入数据解码器和基础数据集类
from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    """数据集分割枚举类"""
    TRAIN = "train"  # 训练集标识
    VAL = "val"      # 验证集标识

    @property
    def dirname(self) -> str:
        """返回对应的目录名称
        
        Returns:
            str: 训练集返回'training'，验证集返回'validation'
        """
        return {
            _Split.TRAIN: "training",
            _Split.VAL: "validation",
        }[self]


def _file_to_segmentation_path(file_name: str, segm_base_path: str) -> str:
    """将图像文件名转换为对应的分割标注文件路径
    
    Args:
        file_name (str): 图像文件名，如'ADE_train_00000001.jpg'
        segm_base_path (str): 分割标注文件的基础路径，通常是'annotations'
        
    Returns:
        str: 分割标注文件的完整路径，如'annotations/ADE_train_00000001.png'
    """
    file_name_noext = os.path.splitext(file_name)[0]  # 移除文件扩展名
    return os.path.join(segm_base_path, file_name_noext + ".png")  # 添加.png扩展名


def _load_segmentation(root: str, split_file_names: List[str]):
    """根据图像文件名列表生成对应的分割标注路径列表
    
    Args:
        root (str): 数据集根目录
        split_file_names (List[str]): 图像文件名列表
        
    Returns:
        List[str]: 分割标注文件的相对路径列表
    """
    segm_base_path = "annotations"  # 标注文件基础目录
    # 为每个图像文件生成对应的分割标注路径
    segmentation_paths = [_file_to_segmentation_path(file_name, segm_base_path) for file_name in split_file_names]
    return segmentation_paths


def _load_file_paths(root: str, split: _Split) -> Tuple[List[str], List[str]]:
    """加载指定数据集分割的文件路径
    
    Args:
        root (str): 数据集根目录路径
        split (_Split): 数据集分割类型（训练集或验证集）
        
    Returns:
        Tuple[List[str], List[str]]: 返回(图像路径列表, 标注路径列表)
    """
    # 读取对应分割的文件列表，格式：ADE20K_object150_train.txt 或 ADE20K_object150_val.txt
    with open(os.path.join(root, f"ADE20K_object150_{split.value}.txt")) as f:
        split_file_names = sorted(f.read().strip().split("\n"))  # 读取并排序文件名

    # 生成分割标注文件路径
    all_segmentation_paths = _load_segmentation(root, split_file_names)
    # 生成图像文件完整路径（添加images前缀）
    file_names = [os.path.join("images", el) for el in split_file_names]
    return file_names, all_segmentation_paths


class ADE20K(ExtendedVisionDataset):
    """ADE20K语义分割数据集
    
    ADE20K是一个用于场景解析的大规模数据集，包含150个语义类别。
    每个像素都有对应的语义标签，适用于语义分割任务。
    
    Attributes:
        Split: 数据集分割类型（训练集/验证集）
        Labels: 标签类型（PIL图像）
    """
    Split = Union[_Split]       # 数据集分割类型定义
    Labels = Union[Image.Image] # 标签数据类型定义

    def __init__(
        self,
        split: "ADE20K.Split",                      # 数据集分割（train/val）
        root: Optional[str] = None,                 # 数据集根目录
        transforms: Optional[Callable] = None,      # 图像和标签的联合变换
        transform: Optional[Callable] = None,       # 图像变换
        target_transform: Optional[Callable] = None,# 标签变换
        image_decoder: Decoder = ImageDataDecoder,  # 图像解码器
        target_decoder: Decoder = DenseTargetDecoder,# 标签解码器（密集标注）
    ) -> None:
        """
        初始化ADE20K数据集
        
        Args:
            split: 数据集分割类型
            root: 数据集根目录路径
            transforms: 应用于图像和标签的联合变换
            transform: 仅应用于图像的变换
            target_transform: 仅应用于标签的变换
            image_decoder: 图像数据解码器
            target_decoder: 标签数据解码器（用于密集分割标注）
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        # 加载图像和标注文件路径
        self.image_paths, self.target_paths = _load_file_paths(root, split)

    def get_image_data(self, index: int) -> bytes:
        """获取指定索引的图像原始数据
        
        Args:
            index (int): 样本索引
            
        Returns:
            bytes: 图像文件的原始字节数据
        """
        image_relpath = self.image_paths[index]  # 获取图像相对路径
        image_full_path = os.path.join(self.root, image_relpath)  # 构造完整路径
        # 以二进制模式读取图像文件
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        """获取指定索引的分割标注数据
        
        Args:
            index (int): 样本索引
            
        Returns:
            Any: 分割标注文件的原始字节数据
        """
        target_relpath = self.target_paths[index]  # 获取标注相对路径
        target_full_path = os.path.join(self.root, target_relpath)  # 构造完整路径
        # 以二进制模式读取标注文件（PNG格式的分割掩码）
        with open(target_full_path, mode="rb") as f:
            target_data = f.read()
        return target_data

    def __len__(self) -> int:
        """返回数据集中的样本数量
        
        Returns:
            int: 数据集样本总数
        """
        return len(self.image_paths)
