# Formal Languages Project

A comprehensive study of transformer models learning formal language recognition tasks, including Parity, Dyck-1, and Dyck-2 languages.

## 📋 Project Overview

This project implements and trains transformer models to recognize and predict properties of formal languages. The focus is on understanding how neural networks learn structured patterns and stack-based computations through sequence-to-sequence learning.

### Supported Tasks

1. **Parity Language**: Predict prefix parity of binary strings
2. **Dyck-1 Language**: Stack prediction for balanced parentheses with reset symbols
3. **Dyck-2 Language**: Enhanced stack prediction for two types of brackets `()` and `[]`

## 🏗️ Architecture

### Softmax Attention Transformer

The project uses a custom `SoftmaxAttentionTransformer` with the following components:

- **Embedding Layer**: Maps vocabulary tokens to dense vectors
- **Positional Encoding**: Sinusoidal encoding for sequence position information
- **Multi-Head Attention**: Parallel attention mechanisms
- **Feed-Forward Networks**: Two-layer MLPs with ReLU activation
- **Layer Normalization**: Applied after attention and feed-forward layers
- **Residual Connections**: Skip connections for stable training

### Model Configurations

| Task | d_model | n_heads | n_layers | Parameters | Output Dim |
|------|---------|---------|----------|------------|------------|
| Parity | 64 | 4 | 2 | ~100K | 2 |
| Dyck-1 | 64 | 4 | 2 | ~100K | 2 |
| Dyck-2 | 64 | 4 | 2 | ~100K | 3 |

## 📁 Project Structure

```
Formal-Languages/
├── data/                          # Data generation and datasets
│   ├── gen_parity.py             # Parity language generation
│   ├── gen_dyck.py               # Dyck-1 language generation
│   ├── gen_dyck2.py              # Dyck-2 language generation
│   ├── parity_dataset.py         # Parity PyTorch datasets
│   ├── dyck_dataset.py           # Dyck-1 PyTorch datasets
│   ├── dyck2_dataset.py          # Dyck-2 PyTorch datasets
│   ├── example_*.py              # Usage examples
│   └── README_*.md               # Task-specific documentation
├── models/                        # Model architectures
│   ├── softmax_attention_transformer.py
│   └── average_hard_attention_transformer.py
├── buildmodels/                   # Training scripts
│   ├── train_parity_simple.py    # Parity training
│   ├── train_dyck1_simple.py     # Dyck-1 training
│   ├── train_dyck2_simple.py     # Dyck-2 training
│   └── RUN_TRAIN_DYCK1.py        # Advanced Dyck-1 training
├── *.pth                         # Trained model files
├── *.png                         # Training curve plots
└── README.md                     # This file
```

## 🎯 Task Descriptions

### 1. Parity Language

**Task**: Predict the parity (even/odd) of the number of 1s in each prefix of a binary string.

**Input**: Binary strings of fixed length (e.g., "10101")
**Output**: One-hot encoded labels for each position
- `[1, 0]` for even number of 1s in prefix
- `[0, 1]` for odd number of 1s in prefix

**Example**:
```
Input:  "10101"
Output: [[0,1], [1,0], [0,1], [1,0], [0,1]]
        (0 1s: even, 1 1: odd, 2 1s: even, 2 1s: even, 3 1s: odd)
```

### 2. Dyck-1 Language (with Reset Symbols)

**Task**: Predict stack state at each position in a string of parentheses with reset symbols.

**Input**: Strings containing `(`, `)`, and `1` (reset symbol)
**Output**: One-hot encoded stack states
- `[1, 0]` for empty stack
- `[0, 1]` for non-empty stack

**Example**:
```
Input:  "(1())"
Output: [[0,1], [1,0], [0,1], [0,1], [1,0]]
        (push, reset, push, push, pop)
```

### 3. Dyck-2 Language

**Task**: Predict enhanced stack state for strings with two types of brackets.

**Input**: Strings containing `(`, `)`, `[`, `]`
**Output**: Three-state one-hot encoding
- `[1, 0, 0]` for empty stack
- `[0, 1, 0]` for round brackets only on stack
- `[0, 0, 1]` for square brackets or mixed brackets on stack

**Example**:
```
Input:  "([)]"
Output: [[0,1,0], [0,0,1], [0,0,1], [1,0,0]]
        (push (, push [ (mixed), pop ], pop )
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy
```

### Data Generation

```python
# Generate parity data
from data.gen_parity import generate_parity_string, generate_prefix_parity_labels_onehot
string = generate_parity_string(10)
labels = generate_prefix_parity_labels_onehot(string)

# Generate Dyck-1 data
from data.gen_dyck import DyckLanguage
dyck = DyckLanguage()
string = dyck.generate(10)
labels = dyck.output_generator(string)

# Generate Dyck-2 data
from data.gen_dyck2 import Dyck2Language
dyck2 = Dyck2Language()
string = dyck2.generate(10)
labels = dyck2.output_generator(string)
```

### Training

```bash
# Train Parity model
cd buildmodels
python train_parity_simple.py

# Train Dyck-1 model
python train_dyck1_simple.py

# Train Dyck-2 model
python train_dyck2_simple.py
```

### Using Trained Models

```python
import torch
from models.softmax_attention_transformer import SoftmaxAttentionTransformer

# Load model
model = SoftmaxAttentionTransformer(vocab_size=4, output_dim=2)
model.load_state_dict(torch.load('parity_model.pth'))

# Make predictions
input_tensor = torch.tensor([[0, 1, 0, 1, 0]])  # "01010"
with torch.no_grad():
    output = model(input_tensor)
    predictions = torch.sigmoid(output) > 0.5
```

## 📊 Results

### Performance Summary

| Task | Test Accuracy | Model Size | Training Time |
|------|---------------|------------|---------------|
| Parity | ~95% | ~100K params | ~5 min |
| Dyck-1 | 99.81% | ~100K params | ~10 min |
| Dyck-2 | 83.82% | ~100K params | ~15 min |

### Key Findings

1. **Model Efficiency**: Medium-sized models (100K parameters) achieve excellent performance across all tasks
2. **Task Complexity**: Dyck-2 is significantly harder than Dyck-1 due to bracket type tracking
3. **Generalization**: All models show good generalization with proper regularization
4. **Learning Rate Scheduling**: Adaptive learning rate scheduling improves convergence

### Training Curves

The training scripts generate plots showing:
- Training and validation loss over time
- Training and validation accuracy over time
- Learning rate changes during training

## 🔧 Configuration

### Model Hyperparameters

```python
# Standard configuration for all tasks
model_config = {
    'd_model': 64,        # Hidden dimension
    'n_heads': 4,         # Number of attention heads
    'n_layers': 2,        # Number of transformer layers
    'dropout': 0.0,       # Dropout rate (disabled for best results)
    'output_dim': 2       # Output dimension (2 for parity/dyck1, 3 for dyck2)
}
```

### Training Hyperparameters

```python
# Training configuration
train_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 130,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5
}
```

### Dataset Configuration

```python
# Dataset sizes
dataset_config = {
    'n_samples': 3000,    # Total samples
    'train_size': 1800,   # Training samples
    'val_size': 600,      # Validation samples
    'test_size': 600      # Test samples
}
```

## 📝 Usage Examples

### Parity Task

```python
from data.example_parity_usage import demonstrate_parity_generation

# Generate and visualize parity data
demonstrate_parity_generation()
```

### Dyck-1 Task

```python
from data.example_dyck_usage import demonstrate_dyck_generation

# Generate and visualize Dyck-1 data
demonstrate_dyck_generation()
```

### Dyck-2 Task

```python
from data.example_dyck2_usage import demonstrate_dyck2_generation

# Generate and visualize Dyck-2 data
demonstrate_dyck2_generation()
```

## 🔍 Analysis Tools

### Overfitting Analysis

```bash
# Analyze Dyck-2 overfitting patterns
python analyze_dyck2_overfitting.py
```

### Dataset Comparison

```bash
# Compare performance across dataset sizes
python compare_dataset_sizes.py
```

## 📚 Documentation

- `data/README_parity.md` - Detailed parity task documentation
- `data/README_dyck.md` - Dyck-1 task documentation  
- `data/README_dyck2.md` - Dyck-2 task documentation
- `PARITY_TRAINING_DETAILS.md` - Parity training specifics
- `SOFTMAX_ATTENTION_TRANSFORMER_ARCHITECTURE.md` - Model architecture details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Based on research in formal language recognition with neural networks
- Inspired by work on transformer architectures for structured tasks
- Built with PyTorch and modern deep learning practices

---

For questions or issues, please open an issue on GitHub.