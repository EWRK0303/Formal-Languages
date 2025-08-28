#!/usr/bin/env python3
"""
Dyck-2 Language Dataset Module
Provide dataset classes and related functions for Dyck-2 language
"""

import torch
from torch.utils.data import Dataset

try:
    from gen_dyck2 import Dyck2Language, generate_dyck2_dataset
except ImportError:
    from .gen_dyck2 import Dyck2Language, generate_dyck2_dataset


class Dyck2Dataset(Dataset):
    """
    Dyck-2 Language Dataset Class
    Maintains consistent interface with other formal language datasets
    """
    
    def __init__(self, data, vocab=None):
        """
        Initialize Dyck-2 dataset
        
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


class StandardDyck2Dataset(Dyck2Dataset):
    """
    Standard Dyck-2 Dataset (only contains valid Dyck-2 strings)
    """
    
    def __init__(self, n_samples=10000, seq_len=16, p=0.4, q=0.3, vocab=None):
        """
        Initialize standard Dyck-2 dataset
        
        Args:
            n_samples: number of samples
            seq_len: sequence length
            p: nesting probability
            q: concatenation probability
            vocab: vocabulary
        """
        data = generate_dyck2_dataset(n_samples, seq_len, p, q)
        super().__init__(data, vocab)


def get_dyck2_dataset(n_samples=10000, seq_len=16, p=0.4, q=0.3, vocab=None):
    """
    Convenient function to get Dyck-2 dataset
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        p: nesting probability
        q: concatenation probability
        vocab: vocabulary
        
    Returns:
        Dyck-2 dataset instance
    """
    return StandardDyck2Dataset(n_samples, seq_len, p, q, vocab)


def test_dyck2_dataset():
    """
    Test Dyck-2 dataset
    """
    print("Testing Dyck-2 dataset...")
    
    # Create small dataset for testing
    dataset = get_dyck2_dataset(n_samples=100, seq_len=8, p=0.4, q=0.3)
    
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
        
        # Verify Dyck-2 validity
        dyck2_gen = Dyck2Language()
        is_valid = dyck2_gen.is_valid_dyck2(x_str)
        print(f"Valid Dyck-2: {is_valid}")
    
    # Count label distribution
    empty_count = 0
    round_only_count = 0
    square_or_mixed_count = 0
    
    for _, y_labels in dataset.data:
        for label in y_labels:
            if label == [1, 0, 0]:  # empty stack
                empty_count += 1
            elif label == [0, 1, 0]:  # round brackets only
                round_only_count += 1
            else:  # square brackets or mixed
                square_or_mixed_count += 1
    
    total = empty_count + round_only_count + square_or_mixed_count
    print(f"\nStack state distribution:")
    print(f"Empty stack: {empty_count} ({empty_count/total:.3f})")
    print(f"Round brackets only: {round_only_count} ({round_only_count/total:.3f})")
    print(f"Square brackets or mixed: {square_or_mixed_count} ({square_or_mixed_count/total:.3f})")


if __name__ == "__main__":
    test_dyck2_dataset() 