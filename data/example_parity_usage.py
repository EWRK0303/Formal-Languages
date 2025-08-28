#!/usr/bin/env python3
"""
Parity Language Usage Examples
Demonstrate how to use data generation modules and dataset classes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gen_parity import (
    generate_parity_string,
    generate_prefix_parity_labels_onehot,
    generate_parity_dataset,
    generate_balanced_parity_dataset,
    verify_parity_labels,
    print_parity_examples
)
from data.parity_dataset import (
    ParityDataset,
    StandardParityDataset,
    BalancedParityDataset,
    get_parity_dataset
)
import torch
from torch.utils.data import DataLoader


def example_basic_generation():
    """Basic data generation example"""
    print("=" * 60)
    print("Basic Data Generation Example")
    print("=" * 60)
    
    # Generate single parity string
    sequence = generate_parity_string(8)
    print(f"Generated binary string: {sequence}")
    
    # Generate prefix parity labels
    labels = generate_prefix_parity_labels_onehot(sequence)
    print(f"Prefix parity labels: {labels}")
    
    # Verify label correctness
    is_correct = verify_parity_labels(sequence, labels)
    print(f"Labels correct: {is_correct}")
    
    # Detailed explanation
    print("\nDetailed explanation:")
    ones_count = 0
    for i, c in enumerate(sequence):
        if c == '1':
            ones_count += 1
        prefix = sequence[:i+1]
        parity = "even" if ones_count % 2 == 0 else "odd"
        print(f"  Position {i}: '{prefix}' → {ones_count} ones → {parity} → {labels[i]}")


def example_dataset_generation():
    """Dataset generation example"""
    print("\n" + "=" * 60)
    print("Dataset Generation Example")
    print("=" * 60)
    
    # Generate standard dataset (only contains strings with even number of 1s)
    print("Generating standard parity dataset...")
    standard_data = generate_parity_dataset(n_samples=100, seq_len=8)
    print(f"Generated {len(standard_data)} samples")
    
    # Verify first few samples
    print("\nVerifying first 3 samples:")
    for i in range(3):
        sequence, labels = standard_data[i]
        is_correct = verify_parity_labels(sequence, labels)
        print(f"Sample {i+1}: {sequence} → Correct: {is_correct}")
    
    # Generate balanced dataset (contains strings with even and odd number of 1s)
    print("\nGenerating balanced parity dataset...")
    balanced_data = generate_balanced_parity_dataset(n_samples=100, seq_len=8)
    print(f"Generated {len(balanced_data)} samples")
    
    # Count label distribution
    even_count = 0
    odd_count = 0
    for _, labels in balanced_data:
        for label in labels:
            if label == [1, 0]:  # even
                even_count += 1
            else:  # odd
                odd_count += 1
    
    print(f"Balanced dataset label distribution:")
    print(f"  Even labels: {even_count}")
    print(f"  Odd labels: {odd_count}")
    print(f"  Ratio: {even_count/(even_count+odd_count):.3f} : {odd_count/(even_count+odd_count):.3f}")


def example_dataset_classes():
    """Dataset class usage example"""
    print("\n" + "=" * 60)
    print("Dataset Class Usage Example")
    print("=" * 60)
    
    # Use standard dataset class
    print("Creating standard parity dataset...")
    standard_dataset = StandardParityDataset(n_samples=100, seq_len=8)
    print(f"Dataset size: {len(standard_dataset)}")
    print(f"Vocabulary: {standard_dataset.vocab}")
    
    # Get a sample
    x_tensor, y_tensor = standard_dataset[0]
    x_str, y_labels = standard_dataset.data[0]
    print(f"\nFirst sample:")
    print(f"  Input string: {x_str}")
    print(f"  Input tensor: {x_tensor}")
    print(f"  Output labels: {y_labels}")
    print(f"  Output tensor: {y_tensor}")
    
    # Use balanced dataset class
    print("\nCreating balanced parity dataset...")
    balanced_dataset = BalancedParityDataset(n_samples=100, seq_len=8)
    print(f"Dataset size: {len(balanced_dataset)}")
    
    # Use convenient function
    print("\nUsing convenient function to create dataset...")
    dataset = get_parity_dataset(n_samples=100, seq_len=8, balanced=True)
    print(f"Dataset size: {len(dataset)}")


def example_data_loading():
    """Data loading example"""
    print("\n" + "=" * 60)
    print("Data Loading Example")
    print("=" * 60)
    
    # Create dataset
    dataset = get_parity_dataset(n_samples=100, seq_len=8, balanced=True)
    
    # Create data loader
    def collate_fn(batch):
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
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Get a batch
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input shape: {x_batch.shape}")
        print(f"  Output shape: {y_batch.shape}")
        print(f"  Input tensor: {x_batch}")
        print(f"  Output tensor: {y_batch}")
        break


def example_parity_examples():
    """Parity examples demonstration"""
    print("\n" + "=" * 60)
    print("Parity Examples Demonstration")
    print("=" * 60)
    
    # Print detailed parity examples
    print_parity_examples(n_examples=3, seq_len=8)


def example_custom_parity():
    """Custom parity examples"""
    print("\n" + "=" * 60)
    print("Custom Parity Examples")
    print("=" * 60)
    
    # Manually create some examples
    test_sequences = [
        "0110",
        "1011", 
        "0000",
        "1111",
        "0101"
    ]
    
    print("Manual parity testing:")
    for sequence in test_sequences:
        labels = generate_prefix_parity_labels_onehot(sequence)
        is_correct = verify_parity_labels(sequence, labels)
        
        print(f"\nInput: {sequence}")
        print(f"Output: {labels}")
        print(f"Correct: {is_correct}")
        
        # Detailed explanation
        ones_count = 0
        for i, c in enumerate(sequence):
            if c == '1':
                ones_count += 1
            prefix = sequence[:i+1]
            parity = "even" if ones_count % 2 == 0 else "odd"
            print(f"  Position {i}: '{prefix}' → {ones_count} ones → {parity} → {labels[i]}")


def main():
    """主函数"""
    print("奇偶校验语言使用示例")
    print("=" * 60)
    
    # 运行所有示例
    example_basic_generation()
    example_dataset_generation()
    example_dataset_classes()
    example_data_loading()
    example_parity_examples()
    example_custom_parity()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main() 