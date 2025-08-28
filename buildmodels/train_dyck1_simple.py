#!/usr/bin/env python3
"""
简化的Dyck-1语言训练脚本
用于快速测试和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

# 添加父目录到路径以导入模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dyck_dataset import get_rdyck_dataset
from models.softmax_attention_transformer import SoftmaxAttentionTransformer


def collate_fn(batch):
    """数据批处理函数"""
    x_batch, y_batch = zip(*batch)
    x_padded = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    
    max_len = max(len(y) for y in y_batch)
    y_padded = []
    for y in y_batch:
        y_tensor = torch.tensor(y, dtype=torch.float)
        if len(y_tensor) < max_len:
            padding = torch.zeros(max_len - len(y_tensor), y_tensor.shape[1])
            y_tensor = torch.cat([y_tensor, padding], dim=0)
        y_padded.append(y_tensor)
    y_padded = torch.stack(y_padded)
    
    return x_padded, y_padded


def train_simple_dyck1():
    """简化的Dyck-1训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成数据集
    print("生成Dyck-1语言数据集...")
    full_dataset = get_rdyck_dataset(
        n_samples=3000,  # 总样本数
        seq_len=16,
        p=0.4,
        q=0.3
    )
    
    # 分割数据集
    train_size = 1800
    val_size = 600
    test_size = 600
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 获取词汇表大小
    vocab_size = len(full_dataset.vocab)
    print(f"词汇表大小: {vocab_size}")
    print(f"词汇表: {full_dataset.vocab}")
    
    # 创建模型 - 与Dyck-2相同的配置
    model = SoftmaxAttentionTransformer(
        vocab_size=vocab_size,
        d_model=64,   # 中等大小
        n_heads=4,    # 中等大小
        n_layers=2,   # 中等大小
        dropout=0.0,  # 禁用dropout
        output_dim=2  # Dyck-1有2个状态
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练设置
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("开始训练...")
    num_epochs = 130  # 与Dyck-2相同的轮数
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算准确率
            predictions = torch.sigmoid(output) > 0.5
            train_correct += (predictions == y).all(dim=2).sum().item()
            train_total += y.size(0) * y.size(1)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                
                predictions = torch.sigmoid(output) > 0.5
                val_correct += (predictions == y).all(dim=2).sum().item()
                val_total += y.size(0) * y.size(1)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 测试阶段
    print("\n测试模型...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            predictions = torch.sigmoid(output) > 0.5
            test_correct += (predictions == y).all(dim=2).sum().item()
            test_total += y.size(0) * y.size(1)
    
    test_accuracy = test_correct / test_total
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dyck1_simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), 'dyck1_model_simple.pth')
    print("模型已保存: dyck1_model_simple.pth")
    
    return model, test_accuracy


def test_model_inference(model, device):
    """测试模型推理"""
    print("\n=== 模型推理测试 ===")
    
    model.eval()
    
    # 测试样本
    test_strings = [
        "()",
        "(1)",
        "(()1)",
        "(1())",
        "((1)1)"
    ]
    
    vocab = {'(': 0, ')': 1, '1': 2}
    
    for test_str in test_strings:
        # 转换为张量
        x = torch.tensor([[vocab[c] for c in test_str]], dtype=torch.long).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(x)
            predictions = torch.sigmoid(output) > 0.5
        
        # 转换为栈状态
        stack_states = []
        for pred in predictions[0]:
            if pred[0] > pred[1]:  # [1, 0] 空栈
                stack_states.append("空栈")
            else:  # [0, 1] 非空栈
                stack_states.append("非空栈")
        
        print(f"输入: {test_str}")
        print(f"预测栈状态: {stack_states}")
        print()


if __name__ == "__main__":
    # 运行简化训练
    model, accuracy = train_simple_dyck1()
    
    # 测试推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model_inference(model, device)
    
    print(f"\n训练完成！最终测试准确率: {accuracy:.4f}") 