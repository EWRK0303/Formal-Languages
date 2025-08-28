# Dyck语言生成器使用指南

本目录包含了用于生成Dyck语言数据集的完整代码，支持标准Dyck语言和重置Dyck-1语言。

## 文件结构

```
data/
├── gen_dyck.py              # Dyck语言生成器核心代码
├── dyck_dataset.py          # PyTorch数据集类
├── generate_dyck_data.py    # 数据生成和保存脚本
├── example_dyck_usage.py    # 使用示例
└── README_dyck.md          # 本说明文档
```

## 核心组件

### 1. DyckLanguage类 (`gen_dyck.py`)

标准Dyck语言生成器，支持多种括号对：

```python
from data.gen_dyck import DyckLanguage

# 创建生成器
dyck_gen = DyckLanguage(num_pairs=2, p=0.4, q=0.3)

# 生成单个样本
sample = dyck_gen.generate(0, 20, 5)

# 生成栈状态输出
output = dyck_gen.output_generator(sample)

# 生成训练集
inputs, outputs, size_info = dyck_gen.training_set_generator(
    num=1000, min_size=5, max_size=50, min_depth=1, max_depth=10
)
```

### 2. RDyck1Language类 (`gen_dyck.py`)

重置Dyck-1语言生成器，在Dyck字符串中插入重置符号：

```python
from data.gen_dyck import RDyck1Language

# 创建生成器
rdyck_gen = RDyck1Language(p=0.4, q=0.3)

# 生成重置Dyck字符串
sample = rdyck_gen.generate_reset_dyck(20)

# 生成栈状态输出
output = rdyck_gen.output_generator(sample)
```

### 3. DyckDataset类 (`dyck_dataset.py`)

PyTorch数据集类，与现有的`FormalLanguageDataset`保持一致的接口：

```python
from data.dyck_dataset import get_dyck_dataset, get_rdyck_dataset

# 获取标准Dyck数据集
dyck_dataset = get_dyck_dataset(
    n_samples=10000, seq_len=16, num_pairs=2, p=0.4, q=0.3
)

# 获取重置Dyck数据集
rdyck_dataset = get_rdyck_dataset(
    n_samples=10000, seq_len=16, p=0.4, q=0.3
)
```

## 快速开始

### 1. 基本使用

```python
# 导入必要的模块
from data.gen_dyck import DyckLanguage
from data.dyck_dataset import get_dyck_dataset
from torch.utils.data import DataLoader

# 生成数据集
dataset = get_dyck_dataset(n_samples=1000, seq_len=16, num_pairs=2)

# 创建数据加载器
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用数据
for x, y in loader:
    # x: 输入序列张量
    # y: 输出标签张量（one-hot编码）
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    break
```

### 2. 生成和保存数据集

```bash
# 生成标准Dyck数据集
python -m data.generate_dyck_data \
    -lang Dyck \
    -num_par 2 \
    -dataset Dyck-2 \
    -training_size 10000 \
    -test_size 2000 \
    -lower_window 5 \
    -upper_window 50

# 生成重置Dyck数据集
python -m data.generate_dyck_data \
    -lang RDyck \
    -dataset RDyck-1 \
    -training_size 10000 \
    -test_size 2000 \
    -lower_window 5 \
    -upper_window 50
```

### 3. 运行示例

```bash
# 运行使用示例
python -m data.example_dyck_usage
```

## 参数说明

### DyckLanguage参数

- `num_pairs`: 使用的括号对数量 (1-7)
  - 支持的括号对: `()`, `[]`, `{}`, `<>`, `+-`, `ab`, `xo`
- `p`: 嵌套概率，控制括号嵌套的深度
- `q`: 连接概率，控制字符串连接
- 约束: `p + q < 1`

### 生成参数

- `min_size/max_size`: 字符串长度范围
- `min_depth/max_depth`: 嵌套深度范围
- `n_samples`: 生成的样本数量

## 输出格式

### 输入格式
Dyck字符串，例如：`"(()())"`, `"[[]()]"`

### 输出格式
每个位置的栈状态（one-hot编码）：
- `[1, 0]`: 空栈
- `[0, 1]`: 非空栈

例如，对于输入 `"(()())"`：
```
位置 0: [0, 1]  # 栈: ['(']
位置 1: [0, 1]  # 栈: ['(', '(']
位置 2: [0, 1]  # 栈: ['(']
位置 3: [1, 0]  # 栈: []
位置 4: [0, 1]  # 栈: ['(']
位置 5: [1, 0]  # 栈: []
```

## 词汇表

### 标准Dyck语言
```python
# 2种括号对
vocab = {
    '(': 0, ')': 1,  # 第一种括号对
    '[': 2, ']': 3   # 第二种括号对
}

# 3种括号对
vocab = {
    '(': 0, ')': 1,  # 第一种括号对
    '[': 2, ']': 3,  # 第二种括号对
    '{': 4, '}': 5   # 第三种括号对
}
```

### 重置Dyck语言
```python
vocab = {
    '(': 0, ')': 1,  # 括号对
    '1': 2           # 重置符号
}
```

## 数据集文件结构

生成的数据集包含以下文件：

```
data/Dyck-2/
├── train_corpus.pk          # 训练语料库（pickle格式）
├── val_corpus_bins.pk       # 验证语料库分箱
├── train_src.txt           # 训练输入序列
├── train_tgt.txt           # 训练输出序列
├── val_src_bin0.txt        # 验证输入序列（箱0）
├── val_tgt_bin0.txt        # 验证输出序列（箱0）
└── data_info.json          # 数据集信息
```

## 与现有代码的集成

### 1. 与FormalLanguageDataset兼容

```python
from data.dataset import FormalLanguageDataset
from data.gen_dyck import generate_dyck_dataset

# 生成数据
data = generate_dyck_dataset(n_samples=1000, seq_len=16)

# 使用现有的数据集类
dataset = FormalLanguageDataset(data)
```

### 2. 与现有模型兼容

```python
from data.dyck_dataset import get_dyck_dataset
from models.softmax_attention_transformer import SoftmaxAttentionTransformer

# 获取数据集
dataset = get_dyck_dataset(n_samples=10000, seq_len=16)

# 创建模型
vocab_size = len(dataset.vocab)
model = SoftmaxAttentionTransformer(
    vocab_size=vocab_size,
    d_model=128,
    nhead=8,
    num_layers=6
)
```

## 测试

运行测试以验证功能：

```bash
# 测试生成器
python -m data.gen_dyck

# 测试数据集
python -m data.dyck_dataset

# 测试完整流程
python -m data.example_dyck_usage
```

## 注意事项

1. **参数设置**: 确保 `p + q < 1`，否则可能无法生成有效的Dyck字符串
2. **内存使用**: 生成大量样本时注意内存使用
3. **序列长度**: 过长的序列可能导致训练困难
4. **词汇表一致性**: 确保训练和测试使用相同的词汇表

## 扩展

### 添加新的括号对

在`DyckLanguage`类中修改`all_pairs`列表：

```python
self.all_pairs = ['()', '[]', '{}', '<>', '+-', 'ab', 'xo', 'yz']  # 添加新的括号对
```

### 自定义输出格式

修改`output_generator`方法以支持不同的输出格式：

```python
def custom_output_generator(self, string):
    # 自定义输出生成逻辑
    pass
```

## 故障排除

### 常见问题

1. **生成速度慢**: 减少`max_depth`或调整`p`、`q`参数
2. **内存不足**: 减少`n_samples`或使用生成器模式
3. **词汇表不匹配**: 确保使用相同的词汇表进行训练和推理

### 调试模式

```python
# 启用调试模式生成小规模数据集
dataset = get_dyck_dataset(n_samples=100, seq_len=8)
``` 