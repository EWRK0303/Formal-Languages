#!/usr/bin/env python3
"""
Parity Language Data Generation Module
Generate binary strings and corresponding prefix parity labels
"""

import random
import numpy as np
from typing import List, Tuple, Dict


def generate_parity_string(seq_len: int) -> str:
    """
    Generate a binary string with even number of 1s
    
    Args:
        seq_len: sequence length
        
    Returns:
        binary string ensuring the entire string has even number of 1s
    """
    # Generate first seq_len-1 random bits
    bits = [random.choice(['0', '1']) for _ in range(seq_len - 1)]
    ones_count = bits.count('1')
    
    # Choose the last bit to ensure the entire string has even number of 1s
    if ones_count % 2 == 0:
        bits.append('0')
    else:
        bits.append('1')
    
    return ''.join(bits)


def generate_prefix_parity_labels_onehot(sequence: str) -> List[List[int]]:
    """
    Generate one-hot labels for prefix parity
    
    Args:
        sequence: binary string
        
    Returns:
        list of one-hot encoded labels, [1,0] for even number of 1s, [0,1] for odd number of 1s
    """
    labels = []
    ones_count = 0
    
    for c in sequence:
        if c == '1':
            ones_count += 1
        
        if ones_count % 2 == 0:
            labels.append([1, 0])  # even number of 1s
        else:
            labels.append([0, 1])  # odd number of 1s
    
    return labels


def generate_parity_dataset(n_samples: int, seq_len: int = 16) -> List[Tuple[str, List[List[int]]]]:
    """
    Generate parity dataset
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        
    Returns:
        dataset list, each element is (input string, output labels)
    """
    dataset = []
    
    for _ in range(n_samples):
        # Generate binary string
        sequence = generate_parity_string(seq_len)
        
        # Generate prefix parity labels
        labels = generate_prefix_parity_labels_onehot(sequence)
        
        dataset.append((sequence, labels))
    
    return dataset


def generate_parity_string_with_odd_parity(seq_len: int) -> str:
    """
    Generate a binary string with odd number of 1s
    
    Args:
        seq_len: sequence length
        
    Returns:
        binary string ensuring the entire string has odd number of 1s
    """
    # Generate first seq_len-1 random bits
    bits = [random.choice(['0', '1']) for _ in range(seq_len - 1)]
    ones_count = bits.count('1')
    
    # Choose the last bit to ensure the entire string has odd number of 1s
    if ones_count % 2 == 0:
        bits.append('1')
    else:
        bits.append('0')
    
    return ''.join(bits)


def generate_balanced_parity_dataset(n_samples: int, seq_len: int = 16) -> List[Tuple[str, List[List[int]]]]:
    """
    Generate balanced parity dataset (containing strings with even and odd number of 1s)
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        
    Returns:
        dataset list, each element is (input string, output labels)
    """
    dataset = []
    half_samples = n_samples // 2
    
    # Generate strings with even number of 1s
    for _ in range(half_samples):
        sequence = generate_parity_string(seq_len)
        labels = generate_prefix_parity_labels_onehot(sequence)
        dataset.append((sequence, labels))
    
    # Generate strings with odd number of 1s
    for _ in range(n_samples - half_samples):
        sequence = generate_parity_string_with_odd_parity(seq_len)
        labels = generate_prefix_parity_labels_onehot(sequence)
        dataset.append((sequence, labels))
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    return dataset


def verify_parity_labels(sequence: str, labels: List[List[int]]) -> bool:
    """
    Verify the correctness of parity labels
    
    Args:
        sequence: binary string
        labels: prefix parity labels
        
    Returns:
        whether the labels are correct
    """
    ones_count = 0
    
    for i, c in enumerate(sequence):
        if c == '1':
            ones_count += 1
        
        expected_label = [1, 0] if ones_count % 2 == 0 else [0, 1]
        
        if labels[i] != expected_label:
            return False
    
    return True


def print_parity_examples(n_examples: int = 3, seq_len: int = 8):
    """
    Print parity examples
    
    Args:
        n_examples: number of examples
        seq_len: sequence length
    """
    print(f"Parity Language Examples (sequence length: {seq_len}):")
    print("=" * 50)
    
    for i in range(n_examples):
        sequence = generate_parity_string(seq_len)
        labels = generate_prefix_parity_labels_onehot(sequence)
        
        print(f"Example {i+1}:")
        print(f"Input: {sequence}")
        print(f"Output: {labels}")
        
        # Verify labels
        is_correct = verify_parity_labels(sequence, labels)
        print(f"Labels correct: {is_correct}")
        
        # Detailed explanation
        print("Detailed explanation:")
        ones_count = 0
        for j, c in enumerate(sequence):
            if c == '1':
                ones_count += 1
            prefix = sequence[:j+1]
            parity = "even" if ones_count % 2 == 0 else "odd"
            print(f"  Position {j}: '{prefix}' → {ones_count} ones → {parity} → {labels[j]}")
        
        print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Print examples
    print_parity_examples()
    
    # Generate dataset
    print("Generating parity dataset...")
    dataset = generate_parity_dataset(1000, seq_len=16)
    print(f"Generated {len(dataset)} samples")
    
    # Verify dataset
    print("Verifying dataset...")
    correct_count = 0
    for sequence, labels in dataset[:10]:  # Only verify first 10 samples
        if verify_parity_labels(sequence, labels):
            correct_count += 1
    
    print(f"First 10 samples: {correct_count}/10 labels correct") 