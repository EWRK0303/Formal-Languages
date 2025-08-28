#!/usr/bin/env python3
"""
Dyck-2 Language Generator
Generate Dyck-2 strings with two types of parentheses: () and []
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any


class Dyck2Language:
    """
    Dyck-2 Language Generator
    Handles two types of parentheses: () and []
    """
    
    def __init__(self, p: float = 0.4, q: float = 0.3):
        """
        Initialize Dyck-2 language generator
        
        Args:
            p: nesting probability
            q: concatenation probability
        """
        self.pairs = ['()', '[]']
        self.p = p
        self.q = q
        
        # Create vocabulary
        self.vocab = {
            '(': 0,  # round opening
            ')': 1,  # round closing
            '[': 2,  # square opening
            ']': 3   # square closing
        }
        
        # Reverse mapping
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
    
    def generate(self, current_size: int = 0, max_size: int = 20, max_depth: int = 5) -> str:
        """
        Generate a single Dyck-2 string
        
        Args:
            current_size: current string length
            max_size: maximum string length
            max_depth: maximum nesting depth
            
        Returns:
            generated Dyck-2 string
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
            # Generate concatenation of two Dyck-2 strings
            first = self.generate(current_size, max_size // 2, max_depth)
            second = self.generate(current_size + len(first), max_size, max_depth)
            return first + second
        
        else:
            # Generate empty string
            return ""
    
    def generate_list(self, num: int = 1000, min_size: int = 5, max_size: int = 50, 
                     min_depth: int = 1, max_depth: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate Dyck-2 string list
        
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
            string: Dyck-2 string
            
        Returns:
            maximum nesting depth
        """
        stack = []
        max_depth = 0
        current_depth = 0
        
        for char in string:
            if char in ['(', '[']:  # opening brackets
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in [')', ']']:  # closing brackets
                current_depth -= 1
        
        return max_depth
    
    def is_valid_dyck2(self, string: str) -> bool:
        """
        Check if string is valid Dyck-2
        
        Args:
            string: string to check
            
        Returns:
            True if valid Dyck-2, False otherwise
        """
        stack = []
        
        for char in string:
            if char in ['(', '[']:  # opening brackets
                stack.append(char)
            elif char in [')', ']']:  # closing brackets
                if not stack:
                    return False
                
                # Check matching
                if char == ')' and stack[-1] != '(':
                    return False
                if char == ']' and stack[-1] != '[':
                    return False
                
                stack.pop()
        
        return len(stack) == 0
    
    def output_generator(self, string: str) -> List[List[int]]:
        """
        Convert Dyck-2 string to enhanced stack state output
        
        Args:
            string: Dyck-2 string
            
        Returns:
            enhanced stack state at each position (3-state encoding)
            [empty, round_only, square_or_mixed]
        """
        stack = []
        outputs = []
        
        for char in string:
            if char in ['(', '[']:  # opening brackets
                stack.append(char)
            elif char in [')', ']']:  # closing brackets
                if stack:
                    # Check matching before popping
                    if char == ')' and stack[-1] == '(':
                        stack.pop()
                    elif char == ']' and stack[-1] == '[':
                        stack.pop()
                    # If not matching, don't pop (invalid string)
            
            # Create enhanced stack state encoding
            if len(stack) == 0:
                outputs.append([1, 0, 0])  # empty stack
            else:
                # Check if stack contains only round brackets
                has_round = any(bracket == '(' for bracket in stack)
                has_square = any(bracket == '[' for bracket in stack)
                
                if has_round and not has_square:
                    outputs.append([0, 1, 0])  # round brackets only
                else:
                    outputs.append([0, 0, 1])  # square brackets or mixed
        
        return outputs
    
    def training_set_generator(self, num: int = 1000, min_size: int = 5, 
                             max_size: int = 50, min_depth: int = 1, 
                             max_depth: int = 10) -> Tuple[List[str], List[List[List[int]]], Dict[str, Any]]:
        """
        Generate training set (input-output pairs)
        
        Args:
            num: number of samples
            min_size: minimum length
            max_size: maximum length
            min_depth: minimum depth
            max_depth: maximum depth
            
        Returns:
            input sequences, output sequences, and size information
        """
        inputs, size_info = self.generate_list(num, min_size, max_size, min_depth, max_depth)
        outputs = [self.output_generator(inp) for inp in inputs]
        
        return inputs, outputs, size_info


def generate_dyck2_dataset(n_samples: int = 10000, seq_len: int = 16, 
                          p: float = 0.4, q: float = 0.3) -> List[Tuple[str, List[List[int]]]]:
    """
    Generate Dyck-2 language dataset
    
    Args:
        n_samples: number of samples
        seq_len: sequence length
        p: nesting probability
        q: concatenation probability
        
    Returns:
        dataset list, each element is (input string, output labels)
    """
    dyck2_gen = Dyck2Language(p=p, q=q)
    data = []
    
    for _ in range(n_samples):
        x = dyck2_gen.generate(0, seq_len, seq_len // 2)
        # Ensure generated string length is close to target
        while len(x) < seq_len // 2:
            x = dyck2_gen.generate(0, seq_len, seq_len // 2)
        
        y = dyck2_gen.output_generator(x)
        data.append((x, y))
    
    return data


def test_dyck2_examples():
    """Test Dyck-2 examples with enhanced stack states"""
    print("=== Dyck-2 Language Examples (Enhanced Stack States) ===")
    print("Stack state encoding: [empty, round_only, square_or_mixed]")
    dyck2_gen = Dyck2Language(p=0.4, q=0.3)
    
    # Test valid examples
    valid_examples = [
        "",
        "()",
        "[]",
        "([()])",
        "[()[]]",
        "()[]",
        "(([]))",
        "[[]()]",
        "((()))",
        "[[[]]]"
    ]
    
    print("\nValid Dyck-2 examples:")
    for example in valid_examples:
        is_valid = dyck2_gen.is_valid_dyck2(example)
        stack_states = dyck2_gen.output_generator(example)
        print(f"  '{example}' -> Valid: {is_valid}")
        print(f"    Stack states: {stack_states}")
    
    # Test invalid examples
    invalid_examples = [
        "([)]",
        "([)",
        "[(])",
        "(",
        ")",
        "[",
        "]"
    ]
    
    print("\nInvalid Dyck-2 examples:")
    for example in invalid_examples:
        is_valid = dyck2_gen.is_valid_dyck2(example)
        stack_states = dyck2_gen.output_generator(example)
        print(f"  '{example}' -> Valid: {is_valid}")
        print(f"    Stack states: {stack_states}")
    
    # Generate random examples
    print("\nGenerated Dyck-2 examples:")
    for i in range(5):
        x = dyck2_gen.generate(0, 12, 4)
        y = dyck2_gen.output_generator(x)
        is_valid = dyck2_gen.is_valid_dyck2(x)
        depth = dyck2_gen.get_depth(x)
        print(f"  Example {i+1}: '{x}' -> Valid: {is_valid}, Depth: {depth}")
        print(f"    Stack states: {y}")
    
    # Demonstrate stack state transitions
    print("\nStack state transitions for '([()])':")
    example = "([()])"
    stack_states = dyck2_gen.output_generator(example)
    print(f"String: {example}")
    for i, (char, state) in enumerate(zip(example, stack_states)):
        state_names = ["empty", "round_only", "square_or_mixed"]
        state_name = state_names[state.index(1)]
        print(f"  Position {i}: '{char}' -> {state} ({state_name})")


if __name__ == "__main__":
    test_dyck2_examples() 