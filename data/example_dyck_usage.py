#!/usr/bin/env python3
"""
Dyck语言使用示例
展示如何使用生成的Dyck语言数据集进行训练
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .dyck_dataset import get_dyck_dataset, get_rdyck_dataset
from .gen_dyck import DyckLanguage, RDyck1Language


def collate_fn(batch):
    """
    数据批处理函数
    处理变长序列的填充
    """
    x_batch, y_batch = zip(*batch)
    
    # 填充输入序列
    x_padded = pad_sequence(x_batch, batch_first=True, padding_value=0)
    
    # 处理输出序列（one-hot编码）
    max_len = max(len(y) for y in y_batch)
    y_padded = []
    
    for y in y_batch:
        y_tensor = torch.tensor(y, dtype=torch.float)
        if len(y_tensor) < max_len:
            # 填充到最大长度
            padding = torch.zeros(max_len - len(y_tensor), y_tensor.shape[1])
            y_tensor = torch.cat([y_tensor, padding], dim=0)
        y_padded.append(y_tensor)
    
    y_padded = torch.stack(y_padded)
    
    return x_padded, y_padded


def example_dyck_training():
    """
    示例：使用Dyck语言数据集进行训练
    """
    print("=== Dyck语言训练示例 ===")
    
    # 1. 生成数据集
    print("1. 生成Dyck语言数据集...")
    train_dataset = get_dyck_dataset(
        n_samples=1000,  # 训练样本数
        seq_len=20,      # 序列长度
        num_pairs=2,     # 使用2种括号对
        p=0.4,           # 嵌套概率
        q=0.3            # 连接概率
    )
    
    # 2. 创建数据加载器
    print("2. 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 3. 查看数据集信息
    print(f"3. 数据集信息:")
    print(f"   训练样本数: {len(train_dataset)}")
    print(f"   词汇表大小: {len(train_dataset.vocab)}")
    print(f"   词汇表: {train_dataset.vocab}")
    
    # 4. 查看几个样本
    print("4. 查看样本:")
    for i, (x, y) in enumerate(train_loader):
        if i >= 2:  # 只显示前2个批次
            break
        print(f"   批次 {i+1}:")
        print(f"     输入形状: {x.shape}")
        print(f"     输出形状: {y.shape}")
        print(f"     输入示例: {x[0]}")
        print(f"     输出示例: {y[0]}")
        print()
    
    return train_loader, train_dataset.vocab


def example_rdyck_training():
    """
    示例：使用重置Dyck语言数据集进行训练
    """
    print("=== 重置Dyck语言训练示例 ===")
    
    # 1. 生成数据集
    print("1. 生成重置Dyck语言数据集...")
    train_dataset = get_rdyck_dataset(
        n_samples=1000,  # 训练样本数
        seq_len=20,      # 序列长度
        p=0.4,           # 嵌套概率
        q=0.3            # 连接概率
    )
    
    # 2. 创建数据加载器
    print("2. 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 3. 查看数据集信息
    print(f"3. 数据集信息:")
    print(f"   训练样本数: {len(train_dataset)}")
    print(f"   词汇表大小: {len(train_dataset.vocab)}")
    print(f"   词汇表: {train_dataset.vocab}")
    
    # 4. 查看几个样本
    print("4. 查看样本:")
    for i, (x, y) in enumerate(train_loader):
        if i >= 2:  # 只显示前2个批次
            break
        print(f"   批次 {i+1}:")
        print(f"     输入形状: {x.shape}")
        print(f"     输出形状: {y.shape}")
        print(f"     输入示例: {x[0]}")
        print(f"     输出示例: {y[0]}")
        print()
    
    return train_loader, train_dataset.vocab


def example_generator_usage():
    """
    示例：直接使用生成器
    """
    print("=== 生成器使用示例 ===")
    
    # 1. 标准Dyck语言生成器
    print("1. 标准Dyck语言生成器:")
    dyck_gen = DyckLanguage(num_pairs=2, p=0.4, q=0.3)
    
    # 生成单个样本
    sample = dyck_gen.generate(0, 15, 5)
    output = dyck_gen.output_generator(sample)
    depth = dyck_gen.get_depth(sample)
    
    print(f"   生成样本: {sample}")
    print(f"   栈状态输出: {output}")
    print(f"   嵌套深度: {depth}")
    print()
    
    # 2. 重置Dyck语言生成器
    print("2. 重置Dyck语言生成器:")
    rdyck_gen = RDyck1Language(p=0.4, q=0.3)
    
    # 生成单个样本
    sample = rdyck_gen.generate_reset_dyck(20)
    output = rdyck_gen.output_generator(sample)
    
    print(f"   生成样本: {sample}")
    print(f"   栈状态输出: {output}")
    print()
    
    # 3. 生成训练集
    print("3. 生成训练集:")
    inputs, outputs, size_info = dyck_gen.training_set_generator(
        num=10,
        min_size=5,
        max_size=15,
        min_depth=1,
        max_depth=5
    )
    
    print(f"   生成样本数: {len(inputs)}")
    print(f"   平均长度: {np.mean(size_info['lengths']):.2f}")
    print(f"   平均深度: {np.mean(size_info['depths']):.2f}")
    print(f"   样本示例:")
    for i in range(3):
        print(f"     {i+1}. 输入: {inputs[i]}")
        print(f"        输出: {outputs[i]}")


def example_custom_vocab():
    """
    示例：使用自定义词汇表
    """
    print("=== 自定义词汇表示例 ===")
    
    # 创建自定义词汇表
    custom_vocab = {
        '(': 0, ')': 1,  # 第一种括号对
        '[': 2, ']': 3,  # 第二种括号对
        'PAD': 4,        # 填充符号
        'UNK': 5         # 未知符号
    }
    
    # 使用自定义词汇表创建数据集
    train_dataset = get_dyck_dataset(
        n_samples=100,
        seq_len=12,
        num_pairs=2,
        p=0.4,
        q=0.3,
        vocab=custom_vocab
    )
    
    print(f"使用自定义词汇表:")
    print(f"   词汇表: {train_dataset.vocab}")
    print(f"   数据集大小: {len(train_dataset)}")
    
    # 查看样本
    x, y = train_dataset[0]
    print(f"   样本输入: {x}")
    print(f"   样本输出: {y}")


if __name__ == "__main__":
    # 运行所有示例
    print("Dyck语言使用示例")
    print("=" * 50)
    
    # 示例1: 标准Dyck语言训练
    example_dyck_training()
    print()
    
    # 示例2: 重置Dyck语言训练
    example_rdyck_training()
    print()
    
    # 示例3: 生成器使用
    example_generator_usage()
    print()
    
    # 示例4: 自定义词汇表
    example_custom_vocab()
    print()
    
    print("所有示例完成！") 