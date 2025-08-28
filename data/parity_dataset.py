#!/usr/bin/env python3
"""
Parity Language Dataset Module
Provide dataset classes and related functions for parity language
"""

import torch
from torch.utils.data import Dataset
try:
    from gen_parity import (
        generate_parity_dataset, 
        generate_balanced_parity_dataset,
        verify_parity_labels
    )
except ImportError:
    from .gen_parity import (
        generate_parity_dataset, 
        generate_balanced_parity_dataset,
        verify_parity_labels
    )


class ParityDataset(Dataset):
    """
    Parity Language Dataset Class
    Maintains consistent interface with FormalLanguageDataset
    """
    
    def __init__(self, data, vocab=None):
        """
        Initialize parity dataset
        
        Args:
            data: data list, each element is (input string, output labels)
            vocab: vocabulary, if None will be created automatically
        """
        self.data = data
        
        # If no vocabulary provided, create from data
        if vocab is None:
            self.vocab = self._create_vocab()
        else:
            self.vocab = vocab
    
    def _create_vocab(self):
        """
        Create vocabulary from data
        
        Returns:
            vocabulary dictionary
        """
        vocab = {}
        char_set = set()
        
        # Collect all unique characters
        for x_str, _ in self.data:
            char_set.update(x_str)
        
        # Create vocabulary mapping
        for i, char in enumerate(sorted(char_set)):
            vocab[char] = i
        
        return vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_str, y = self.data[idx]
        
        # Convert input string to tensor
        x_tensor = torch.tensor([self.vocab[c] for c in x_str], dtype=torch.long)
        
        # Convert output labels to tensor
        if isinstance(y, str):
            y_tensor = torch.tensor([self.vocab[c] for c in y], dtype=torch.long)
        else:
            # y is a list of one-hot encoded labels
            y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor


class StandardParityDataset(ParityDataset):
    """
    Standard Parity Dataset (only contains strings with even number of 1s)
    """
    
    def __init__(self, n_samples=10000, seq_len=16, vocab=None):
        """
        Initialize standard parity dataset
        
        Args:
            n_samples: number of samples
            seq_len: sequence length
            vocab: vocabulary
        """
        data = generate_parity_dataset(n_samples, seq_len)
        super().__init__(data, vocab)


class BalancedParityDataset(ParityDataset):
    """
    Balanced Parity Dataset (contains strings with even and odd number of 1s)
    """
    
    def __init__(self, n_samples=10000, seq_len=16, vocab=None):
        """
        Initialize balanced parity dataset
        
        Args:
            n_samples: number of samples
            seq_len: sequence length
            vocab: vocabulary
        """
        data = generate_balanced_parity_dataset(n_samples, seq_len)
        super().__init__(data, vocab)


def get_parity_dataset(n_samples=10000, seq_len=16, balanced=True, vocab=None):
    """
    Convenient function to get parity dataset
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        balanced: whether to use balanced dataset
        vocab: vocabulary
        
    Returns:
        parity dataset instance
    """
    if balanced:
        return BalancedParityDataset(n_samples, seq_len, vocab)
    else:
        return StandardParityDataset(n_samples, seq_len, vocab)


def test_parity_dataset():
    """
    Test parity dataset
    """
    print("Testing parity dataset...")
    
    # Create small dataset for testing
    dataset = get_parity_dataset(n_samples=100, seq_len=8, balanced=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary: {dataset.vocab}")
    
    # Test several samples
    for i in range(3):
        x_tensor, y_tensor = dataset[i]
        x_str, y_labels = dataset.data[i]
        
        print(f"\nSample {i+1}:")
        print(f"Input string: {x_str}")
        print(f"Input tensor: {x_tensor}")
        print(f"Output labels: {y_labels}")
        print(f"Output tensor: {y_tensor}")
        
        # Verify label correctness
        is_correct = verify_parity_labels(x_str, y_labels)
        print(f"Labels correct: {is_correct}")
    
    # Count label distribution
    even_count = 0
    odd_count = 0
    
    for _, y_labels in dataset.data:
        for label in y_labels:
            if label == [1, 0]:  # even
                even_count += 1
            else:  # odd
                odd_count += 1
    
    print(f"\nLabel distribution:")
    print(f"Even labels: {even_count}")
    print(f"Odd labels: {odd_count}")
    print(f"Ratio: {even_count/(even_count+odd_count):.3f} : {odd_count/(even_count+odd_count):.3f}")


if __name__ == "__main__":
    test_parity_dataset() 