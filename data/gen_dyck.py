#!/usr/bin/env python3
"""
Dyck Language Generator
Generate Dyck-1 strings with parentheses and reset symbols
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any


class DyckLanguage:
    """
    Dyck Language Generator
    Handles standard Dyck-1 language with parentheses
    """
    
    def __init__(self, p: float = 0.4, q: float = 0.3):
        """
        Initialize Dyck language generator
        
        Args:
            p: nesting probability
            q: concatenation probability
        """
        self.pairs = ['()']
        self.p = p
        self.q = q
        
        # Create vocabulary
        self.vocab = {
            '(': 0,  # opening parenthesis
            ')': 1   # closing parenthesis
        }
        
        # Reverse mapping
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
    
    def generate(self, current_size: int = 0, max_size: int = 20, max_depth: int = 5) -> str:
        """
        Generate a single Dyck string
        
        Args:
            current_size: current string length
            max_size: maximum string length
            max_depth: maximum nesting depth
            
        Returns:
            generated Dyck string
        """
        if current_size >= max_size:
            return ""
        
        # Choose generation rule
        rand = random.random()
        
        if rand < self.p and max_depth > 0:
            # Generate nested parentheses
            pair = random.choice(self.pairs)
            inner = self.generate(current_size + 2, max_size, max_depth - 1)
            return pair[0] + inner + pair[1]
        
        elif rand < self.p + self.q and current_size < max_size - 1:
            # Generate concatenation of two Dyck strings
            first = self.generate(current_size, max_size // 2, max_depth)
            second = self.generate(current_size + len(first), max_size, max_depth)
            return first + second
        
        else:
            # Generate empty string
            return ""
    
    def generate_list(self, num: int = 1000, min_size: int = 5, max_size: int = 50, 
                     min_depth: int = 1, max_depth: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate Dyck string list
        
        Args:
            num: number of samples
            min_size: minimum length
            max_size: maximum length
            min_depth: minimum depth
            max_depth: maximum depth
            
        Returns:
            string list and size information
        """
        samples = []
        size_info = {
            'lengths': [],
            'depths': []
        }
        
        for _ in range(num):
            sample = self.generate(0, max_size, max_depth)
            depth = self.get_depth(sample)
            
            # Ensure length and depth requirements
            while len(sample) < min_size or depth < min_depth:
                sample = self.generate(0, max_size, max_depth)
                depth = self.get_depth(sample)
            
            samples.append(sample)
            size_info['lengths'].append(len(sample))
            size_info['depths'].append(depth)
        
        return samples, size_info
    
    def get_depth(self, string: str) -> int:
        """
        Calculate maximum nesting depth
        
        Args:
            string: Dyck string
            
        Returns:
            maximum nesting depth
        """
        stack = []
        max_depth = 0
        current_depth = 0
        
        for char in string:
            if char == '(':  # opening bracket
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':  # closing bracket
                current_depth -= 1
        
        return max_depth
    
    def is_valid_dyck(self, string: str) -> bool:
        """
        Check if string is valid Dyck
        
        Args:
            string: string to check
            
        Returns:
            True if valid Dyck, False otherwise
        """
        stack = []
        
        for char in string:
            if char == '(':  # opening bracket
                stack.append(char)
            elif char == ')':  # closing bracket
                if not stack:
                    return False
                stack.pop()
        
        return len(stack) == 0
    
    def output_generator(self, string: str) -> List[List[int]]:
        """
        Convert Dyck string to stack state output
        
        Args:
            string: Dyck string
            
        Returns:
            stack state at each position (one-hot encoding)
        """
        stack = []
        outputs = []
        
        for char in string:
            if char == '(':  # opening bracket
                stack.append(char)
            elif char == ')':  # closing bracket
                if stack:
                    stack.pop()
            
            # Create stack state one-hot encoding
            if len(stack) == 0:
                outputs.append([1, 0])  # empty stack
            else:
                outputs.append([0, 1])  # non-empty stack
        
        return outputs


class RDyck1Language:
    """
    Reset Dyck-1 Language Generator
    Handles Dyck-1 with reset symbols ('1')
    """
    
    def __init__(self, p: float = 0.4, q: float = 0.3):
        """
        Initialize reset Dyck-1 language generator
        
        Args:
            p: nesting probability
            q: concatenation probability
        """
        self.p = p
        self.q = q
        
        # Create vocabulary
        self.vocab = {
            '(': 0,  # opening parenthesis
            ')': 1,  # closing parenthesis
            '1': 2   # reset symbol
        }
        
        # Reverse mapping
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
    
    def generate(self, current_size: int = 0, max_size: int = 20, max_depth: int = 5) -> str:
        """
        Generate a single reset Dyck string
        
        Args:
            current_size: current string length
            max_size: maximum string length
            max_depth: maximum nesting depth
            
        Returns:
            generated reset Dyck string
        """
        if current_size >= max_size:
            return ""
        
        # Choose generation rule
        rand = random.random()
        
        if rand < self.p and max_depth > 0:
            # Generate nested parentheses
            inner = self.generate(current_size + 2, max_size, max_depth - 1)
            return '(' + inner + ')'
        
        elif rand < self.p + self.q and current_size < max_size - 1:
            # Generate concatenation
            first = self.generate(current_size, max_size // 2, max_depth)
            second = self.generate(current_size + len(first), max_size, max_depth)
            return first + second
        
        elif rand < self.p + self.q + 0.1:  # 10% chance for reset symbol
            # Generate reset symbol
            return '1'
        
        else:
            # Generate empty string
            return ""
    
    def generate_list(self, num: int = 1000, min_size: int = 5, max_size: int = 50, 
                     min_depth: int = 1, max_depth: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate reset Dyck string list
        
        Args:
            num: number of samples
            min_size: minimum length
            max_size: maximum length
            min_depth: minimum depth
            max_depth: maximum depth
            
        Returns:
            string list and size information
        """
        samples = []
        size_info = {
            'lengths': [],
            'depths': []
        }
        
        for _ in range(num):
            sample = self.generate(0, max_size, max_depth)
            depth = self.get_depth(sample)
            
            # Ensure length and depth requirements
            while len(sample) < min_size or depth < min_depth:
                sample = self.generate(0, max_size, max_depth)
                depth = self.get_depth(sample)
            
            samples.append(sample)
            size_info['lengths'].append(len(sample))
            size_info['depths'].append(depth)
        
        return samples, size_info
    
    def get_depth(self, string: str) -> int:
        """
        Calculate maximum nesting depth
        
        Args:
            string: reset Dyck string
            
        Returns:
            maximum nesting depth
        """
        stack = []
        max_depth = 0
        current_depth = 0
        
        for char in string:
            if char == '(':  # opening bracket
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':  # closing bracket
                current_depth -= 1
            elif char == '1':  # reset symbol
                current_depth = 0  # reset depth
        
        return max_depth
    
    def is_valid_rdyck(self, string: str) -> bool:
        """
        Check if string is valid reset Dyck
        
        Args:
            string: string to check
            
        Returns:
            True if valid reset Dyck, False otherwise
        """
        stack = []
        
        for char in string:
            if char == '(':  # opening bracket
                stack.append(char)
            elif char == ')':  # closing bracket
                if not stack:
                    return False
                stack.pop()
            elif char == '1':  # reset symbol
                stack.clear()  # clear stack
        
        return True  # Reset Dyck allows any sequence
    
    def output_generator(self, string: str) -> List[List[int]]:
        """
        Convert reset Dyck string to stack state output
        
        Args:
            string: reset Dyck string
            
        Returns:
            stack state at each position (one-hot encoding)
        """
        stack = []
        outputs = []
        
        for char in string:
            if char == '(':  # opening bracket
                stack.append(char)
            elif char == ')':  # closing bracket
                if stack:
                    stack.pop()
            elif char == '1':  # reset symbol
                stack.clear()  # clear stack
            
            # Create stack state one-hot encoding
            if len(stack) == 0:
                outputs.append([1, 0])  # empty stack
            else:
                outputs.append([0, 1])  # non-empty stack
        
        return outputs


def generate_dyck_dataset(n_samples: int = 10000, seq_len: int = 16, 
                         num_pairs: int = 1, p: float = 0.4, q: float = 0.3) -> List[Tuple[str, List[List[int]]]]:
    """
    Generate Dyck language dataset
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        num_pairs: number of bracket pairs
        p: nesting probability
        q: concatenation probability
        
    Returns:
        dataset list, each element is (input string, output labels)
    """
    dyck_gen = DyckLanguage(p=p, q=q)
    data = []
    
    for _ in range(n_samples):
        x = dyck_gen.generate(0, seq_len, seq_len // 2)
        # Ensure generated string length is close to target
        while len(x) < seq_len // 2:
            x = dyck_gen.generate(0, seq_len, seq_len // 2)
        
        y = dyck_gen.output_generator(x)
        data.append((x, y))
    
    return data


def generate_rdyck_dataset(n_samples: int = 10000, seq_len: int = 16, 
                          p: float = 0.4, q: float = 0.3) -> List[Tuple[str, List[List[int]]]]:
    """
    Generate reset Dyck language dataset
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        p: nesting probability
        q: concatenation probability
        
    Returns:
        dataset list, each element is (input string, output labels)
    """
    rdyck_gen = RDyck1Language(p=p, q=q)
    data = []
    
    for _ in range(n_samples):
        x = rdyck_gen.generate(0, seq_len, seq_len // 2)
        # Ensure generated string length is close to target
        while len(x) < seq_len // 2:
            x = rdyck_gen.generate(0, seq_len, seq_len // 2)
        
        y = rdyck_gen.output_generator(x)
        data.append((x, y))
    
    return data


def test_dyck_examples():
    """Test Dyck examples"""
    print("=== Dyck Language Examples ===")
    dyck_gen = DyckLanguage(p=0.4, q=0.3)
    
    # Test valid examples
    valid_examples = [
        "",
        "()",
        "(())",
        "()()",
        "((()))",
        "(()())"
    ]
    
    print("Valid Dyck examples:")
    for example in valid_examples:
        is_valid = dyck_gen.is_valid_dyck(example)
        stack_states = dyck_gen.output_generator(example)
        print(f"  '{example}' -> Valid: {is_valid}, Stack states: {stack_states}")
    
    # Test invalid examples
    invalid_examples = [
        "(",
        ")",
        ")(",
        "(()",
        "())"
    ]
    
    print("\nInvalid Dyck examples:")
    for example in invalid_examples:
        is_valid = dyck_gen.is_valid_dyck(example)
        stack_states = dyck_gen.output_generator(example)
        print(f"  '{example}' -> Valid: {is_valid}, Stack states: {stack_states}")
    
    # Test reset Dyck examples
    print("\n=== Reset Dyck Language Examples ===")
    rdyck_gen = RDyck1Language(p=0.4, q=0.3)
    
    reset_examples = [
        "",
        "()",
        "1",
        "()1",
        "1()",
        "(1)",
        "()1()"
    ]
    
    print("Reset Dyck examples:")
    for example in reset_examples:
        is_valid = rdyck_gen.is_valid_rdyck(example)
        stack_states = rdyck_gen.output_generator(example)
        print(f"  '{example}' -> Valid: {is_valid}, Stack states: {stack_states}")


if __name__ == "__main__":
    test_dyck_examples() 