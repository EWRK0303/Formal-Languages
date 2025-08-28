# 奇偶校验语言实现

## 概述

奇偶校验语言是一个基础的序列到序列学习任务，我们训练Transformer模型来预测二进制字符串中每个前缀的奇偶校验（1的个数的奇偶性）。这个任务作为测试模型计数和状态跟踪能力的基准。

## 任务定义

### 输入
- **二进制字符串**：固定长度（通常16位）
- **词汇表**：{'0': 0, '1': 1}（2个符号）
- **示例输入**："0110"

### 输出
- **前缀奇偶校验标签**：序列中每个位置的标签
- **格式**：One-hot编码标签 [1,0] 表示偶数，[0,1] 表示奇数
- **示例输出**：[[1,0], [0,1], [1,0], [1,0]]

### 任务规则
对于序列中的每个位置i，预测前缀（位置0到i）中1的个数是偶数还是奇数：
- **偶数个1**：输出 [1,0]
- **奇数个1**：输出 [0,1]

## 实现文件

### 数据生成模块
- `gen_parity.py`：核心数据生成函数
  - `generate_parity_string()`：生成具有偶数个1的二进制字符串
  - `generate_prefix_parity_labels_onehot()`：生成前缀奇偶校验标签
  - `generate_parity_dataset()`：生成标准数据集
  - `generate_balanced_parity_dataset()`：生成平衡数据集
  - `verify_parity_labels()`：验证标签正确性

### 数据集类
- `parity_dataset.py`：数据集类实现
  - `ParityDataset`：基础奇偶校验数据集类
  - `StandardParityDataset`：标准数据集（只包含偶数个1的字符串）
  - `BalancedParityDataset`：平衡数据集（包含偶数个1和奇数个1的字符串）
  - `get_parity_dataset()`：便捷函数

### 训练脚本
- `buildmodels/train_parity.py`：完整训练脚本
- `buildmodels/train_parity_simple.py`：简化训练脚本（用于快速测试）

### 使用示例
- `example_parity_usage.py`：详细的使用示例

## 示例

### 示例1
```
输入: "0110"
输出: [[1,0], [0,1], [1,0], [1,0]]
解释:
位置 0: "0" → 0个1 → 偶数 → [1,0]
位置 1: "01" → 1个1 → 奇数 → [0,1]
位置 2: "011" → 2个1 → 偶数 → [1,0]
位置 3: "0110" → 2个1 → 偶数 → [1,0]
```

### 示例2
```
输入: "1011"
输出: [[0,1], [1,0], [0,1], [1,0]]
解释:
位置 0: "1" → 1个1 → 奇数 → [0,1]
位置 1: "10" → 1个1 → 奇数 → [0,1]
位置 2: "101" → 2个1 → 偶数 → [1,0]
位置 3: "1011" → 3个1 → 奇数 → [0,1]
```

## 使用方法

### 1. 基本数据生成
```python
from data.gen_parity import generate_parity_string, generate_prefix_parity_labels_onehot

# 生成二进制字符串
sequence = generate_parity_string(8)
print(f"生成的字符串: {sequence}")

# 生成标签
labels = generate_prefix_parity_labels_onehot(sequence)
print(f"前缀奇偶校验标签: {labels}")
```

### 2. 创建数据集
```python
from data.parity_dataset import get_parity_dataset

# 创建平衡数据集
dataset = get_parity_dataset(n_samples=1000, seq_len=16, balanced=True)

# 获取样本
x_tensor, y_tensor = dataset[0]
print(f"输入张量: {x_tensor}")
print(f"输出张量: {y_tensor}")
```

### 3. 训练模型
```python
# 运行简化训练脚本
python buildmodels/train_parity_simple.py

# 运行完整训练脚本
python buildmodels/train_parity.py
```

### 4. 查看示例
```python
# 运行使用示例
python data/example_parity_usage.py
```

## 数据生成算法

### 奇偶校验字符串生成
```python
def generate_parity_string(seq_len):
    """生成具有偶数个1的二进制字符串"""
    bits = [random.choice(['0', '1']) for _ in range(seq_len - 1)]
    ones_count = bits.count('1')
    
    # 选择最后一位以确保整个字符串有偶数个1
    if ones_count % 2 == 0:
        bits.append('0')
    else:
        bits.append('1')
    
    return ''.join(bits)
```

**关键特性：**
- 生成随机二进制字符串
- 确保整个字符串有偶数个1
- 最后一位的选择用于维持偶数奇偶校验

### 标签生成
```python
def generate_prefix_parity_labels_onehot(sequence):
    """为前缀奇偶校验生成one-hot标签"""
    labels = []
    ones_count = 0
    
    for c in sequence:
        if c == '1':
            ones_count += 1
        
        if ones_count % 2 == 0:
            labels.append([1, 0])  # 偶数个1
        else:
            labels.append([0, 1])  # 奇数个1
    
    return labels
```

**过程：**
1. 初始化1的计数器
2. 对于每个位置，计算前缀中1的个数
3. 如果个数是偶数输出[1,0]，否则输出[0,1]

## 模型训练

### 模型架构
使用SoftmaxAttentionTransformer模型：
- 词汇表大小：2（'0'和'1'）
- 输出维度：2（one-hot编码）
- 损失函数：BCEWithLogitsLoss
- 优化器：Adam

### 训练参数
- 序列长度：16位
- 批次大小：32
- 学习率：0.001
- 训练轮数：20
- 数据集大小：10,000样本

### 评估指标
- 准确率：预测标签与真实标签的匹配程度
- 损失：二元交叉熵损失

## 验证和测试

### 标签验证
```python
from data.gen_parity import verify_parity_labels

# 验证标签正确性
is_correct = verify_parity_labels(sequence, labels)
print(f"标签正确: {is_correct}")
```

### 数据集测试
```python
from data.parity_dataset import test_parity_dataset

# 测试数据集
test_parity_dataset()
```

## 文件结构

```
data/
├── gen_parity.py              # 数据生成核心函数
├── parity_dataset.py          # 数据集类实现
├── example_parity_usage.py    # 使用示例
└── README_parity.md          # 本文档

buildmodels/
├── train_parity.py           # 完整训练脚本
└── train_parity_simple.py    # 简化训练脚本
```

## 扩展和修改

### 修改序列长度
```python
# 在数据生成时指定序列长度
dataset = get_parity_dataset(n_samples=1000, seq_len=32)
```

### 修改数据集平衡
```python
# 使用不平衡数据集（只包含偶数个1的字符串）
dataset = get_parity_dataset(n_samples=1000, balanced=False)
```

### 自定义词汇表
```python
# 使用自定义词汇表
custom_vocab = {'0': 0, '1': 1}
dataset = get_parity_dataset(n_samples=1000, vocab=custom_vocab)
```

## 注意事项

1. **数据平衡**：平衡数据集包含相等数量的偶数个1和奇数个1的字符串，有助于模型学习
2. **序列长度**：较长的序列会增加任务难度，需要更大的模型和更多的训练数据
3. **验证**：始终验证生成的标签正确性，确保数据质量
4. **模型大小**：根据序列长度和任务复杂度调整模型参数

## 相关研究

奇偶校验任务常用于：
- 测试神经网络的计数能力
- 评估序列模型的长期依赖学习
- 研究Transformer的注意力机制
- 比较不同架构的性能

这个实现提供了一个完整的框架来研究和实验奇偶校验语言学习任务。 