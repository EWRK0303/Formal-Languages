# Dyck-2 Language Task

## Overview

The Dyck-2 language task extends the Dyck-1 concept to handle two types of parentheses: round brackets `()` and square brackets `[]`. The model must track the enhanced stack state while ensuring proper nesting and matching of both types of parentheses.

## Task Definition

### Input
- **Dyck-2 strings** (balanced parentheses with two types) of variable length
- **Vocabulary**: {'(': 0, ')': 1, '[': 2, ']': 3} (4 symbols)
- **Example input**: "([()])"

### Output
- **Enhanced stack state predictions** for each position in the sequence
- **Format**: 3-state one-hot encoded labels [empty, round_only, square_or_mixed]
- **Example output**: [[0,1,0], [0,0,1], [0,1,0], [0,1,0], [0,0,1], [1,0,0]]

### Task Rules

For each position in the sequence, predict the enhanced stack state after processing that symbol:

1. **'(' or '['**: Push to stack
2. **')' or ']'**: Pop from stack (if matching)
3. **Stack state classification**:
   - **[1,0,0]**: Empty stack
   - **[0,1,0]**: Stack contains only round brackets `()`
   - **[0,0,1]**: Stack contains square brackets `[]` or mixed types

### Valid Dyck-2 Examples

**Simple cases:**
- `""` → `[]` (empty)
- `"()"` → `[[0,1,0], [1,0,0]]`
- `"[]"` → `[[0,0,1], [1,0,0]]`

**Nested cases:**
- `"([()])"` → `[[0,0,1], [0,1,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]]`
- `"[()[]]"` → `[[0,0,1], [0,1,0], [0,1,0], [0,0,1], [0,0,1], [1,0,0]]`

**Concatenated cases:**
- `"()[]"` → `[[0,1,0], [1,0,0], [0,0,1], [1,0,0]]`

### Invalid Examples

- `"([)]"` - Wrong nesting order
- `"([)"` - Unmatched brackets
- `"[(])"` - Crossed brackets
- `"("` - Unclosed bracket

## Implementation Details

### Data Generation

The Dyck-2 language generator uses a probabilistic grammar with three rules:

1. **Nesting** (probability p): Generate `(inner)` or `[inner]`
2. **Concatenation** (probability q): Generate `first + second`
3. **Empty** (probability 1-p-q): Generate empty string

### Enhanced Stack State Tracking

Unlike Dyck-1 which only tracks empty/non-empty states, Dyck-2 tracks:

1. **Empty stack** `[1,0,0]`: No brackets on stack
2. **Round brackets only** `[0,1,0]`: Stack contains only `(` brackets
3. **Square brackets or mixed** `[0,0,1]`: Stack contains `[` brackets or mixed types

This distinction is crucial because:
- It captures the different bracket types
- It provides more informative state representation
- It enables better understanding of the model's learning

### Model Architecture

- **Input**: Dyck-2 strings with 4-symbol vocabulary
- **Output**: 3-state stack predictions for each position
- **Loss**: BCEWithLogitsLoss for multi-label classification
- **Architecture**: SoftmaxAttentionTransformer

## Key Differences from Dyck-1

| Aspect | Dyck-1 | Dyck-2 |
|--------|--------|--------|
| Vocabulary | 2 symbols | 4 symbols |
| Stack states | 2 states (empty/non-empty) | 3 states (empty/round_only/square_or_mixed) |
| Complexity | Linear | Higher (two bracket types) |
| Nesting rules | Single type | Two types with matching rules |

## Research Applications

1. **Hierarchical Structure Learning**: Understanding how models learn nested structures
2. **Multi-Type Pattern Recognition**: Testing ability to distinguish between different bracket types
3. **Enhanced State Tracking**: Evaluating models' capacity for detailed state representation
4. **Formal Language Theory**: Investigating context-free language recognition capabilities

## Theoretical Significance

- **Context-Free Grammar**: Dyck-2 is a classic example of a context-free language
- **Stack Automata**: Requires a pushdown automaton with enhanced state tracking
- **Hierarchical Parsing**: Tests models' ability to parse nested structures with multiple types
- **State Complexity**: Demonstrates the need for more sophisticated state representations

## Usage

### Basic Usage

```python
from data.dyck2_dataset import get_dyck2_dataset

# Generate dataset
dataset = get_dyck2_dataset(n_samples=1000, seq_len=16)

# Access data
x_tensor, y_tensor = dataset[0]
print(f"Input: {x_tensor}")
print(f"Output: {y_tensor}")
```

### Training

```python
from buildmodels.train_dyck2_simple import train_simple_dyck2

# Train model
model, train_losses, val_losses, train_accuracies, val_accuracies = train_simple_dyck2()
```

### Testing Examples

```python
from data.gen_dyck2 import Dyck2Language

dyck2_gen = Dyck2Language()
example = "([()])"
stack_states = dyck2_gen.output_generator(example)
print(f"Stack states: {stack_states}")
```

## File Structure

```
data/
├── gen_dyck2.py              # Dyck-2 language generator
├── dyck2_dataset.py          # PyTorch dataset classes
└── README_dyck2.md          # This documentation

buildmodels/
└── train_dyck2_simple.py    # Training script
```

## Evaluation Metrics

1. **Accuracy**: Percentage of correctly predicted stack states
2. **State Distribution**: Balance between empty, round-only, and square/mixed states
3. **Length Generalization**: Performance on longer sequences than training
4. **Depth Generalization**: Performance on deeper nesting than training

## Challenges

1. **State Complexity**: 3-state output requires more sophisticated learning
2. **Bracket Type Distinction**: Model must learn to distinguish between `()` and `[]`
3. **Mixed State Handling**: Complex cases with both bracket types on stack
4. **Nesting Depth**: Deeper nesting increases difficulty
5. **Cross-Sequence Generalization**: Performance on unseen bracket patterns 