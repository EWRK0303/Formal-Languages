#!/usr/bin/env python3
"""
Dyck-2 Language Usage Examples
Demonstrate various Dyck-2 generation and usage patterns
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gen_dyck2 import Dyck2Language, generate_dyck2_dataset


def demonstrate_dyck2_generation():
    """Demonstrate Dyck-2 string generation with different parameters"""
    
    print("=== Dyck-2 Generation Examples ===")
    
    # Create generator with different parameters
    dyck2_gen = Dyck2Language(p=0.4, q=0.3)
    
    print("\n1. Simple Dyck-2 strings (max_length=8):")
    for i in range(5):
        string = dyck2_gen.generate(0, 8, 3)
        depth = dyck2_gen.get_depth(string)
        is_valid = dyck2_gen.is_valid_dyck2(string)
        print(f"  {i+1}. '{string}' -> Valid: {is_valid}, Depth: {depth}")
    
    print("\n2. Medium complexity Dyck-2 strings (max_length=12):")
    for i in range(5):
        string = dyck2_gen.generate(0, 12, 5)
        depth = dyck2_gen.get_depth(string)
        is_valid = dyck2_gen.is_valid_dyck2(string)
        print(f"  {i+1}. '{string}' -> Valid: {is_valid}, Depth: {depth}")
    
    print("\n3. Complex Dyck-2 strings (max_length=16):")
    for i in range(5):
        string = dyck2_gen.generate(0, 16, 7)
        depth = dyck2_gen.get_depth(string)
        is_valid = dyck2_gen.is_valid_dyck2(string)
        print(f"  {i+1}. '{string}' -> Valid: {is_valid}, Depth: {depth}")


def demonstrate_stack_states():
    """Demonstrate stack state tracking for various Dyck-2 strings"""
    
    print("\n=== Stack State Tracking Examples ===")
    
    dyck2_gen = Dyck2Language()
    
    # Test cases with different patterns
    test_cases = [
        "()",           # Simple round brackets
        "[]",           # Simple square brackets
        "()[]",         # Concatenated
        "([()])",       # Nested mixed
        "[()[]]",       # Complex nested
        "(([]))",       # Round with square inside
        "[[[]]]",       # Deep square nesting
        "()()()",       # Multiple round pairs
        "[][][]",       # Multiple square pairs
        "([()])[()]",   # Complex mixed
    ]
    
    for i, test_string in enumerate(test_cases):
        print(f"\n{i+1}. String: '{test_string}'")
        stack_states = dyck2_gen.output_generator(test_string)
        is_valid = dyck2_gen.is_valid_dyck2(test_string)
        depth = dyck2_gen.get_depth(test_string)
        
        print(f"   Valid: {is_valid}, Depth: {depth}")
        print(f"   Stack states: {stack_states}")
        
        # Show state transitions
        print("   State transitions:")
        for j, (char, state) in enumerate(zip(test_string, stack_states)):
            state_names = ["empty", "round_only", "square_or_mixed"]
            state_name = state_names[state.index(1)]
            print(f"     Position {j}: '{char}' -> {state} ({state_name})")


def demonstrate_dataset_generation():
    """Demonstrate dataset generation and statistics"""
    
    print("\n=== Dataset Generation Examples ===")
    
    # Generate small dataset for demonstration
    print("Generating Dyck-2 dataset (100 samples, max_length=10)...")
    data = generate_dyck2_dataset(n_samples=100, seq_len=10, p=0.4, q=0.3)
    
    print(f"Generated {len(data)} samples")
    
    # Show some examples from the dataset
    print("\nSample dataset entries:")
    for i in range(5):
        x_str, y_labels = data[i]
        is_valid = Dyck2Language().is_valid_dyck2(x_str)
        depth = Dyck2Language().get_depth(x_str)
        print(f"  {i+1}. Input: '{x_str}'")
        print(f"     Valid: {is_valid}, Depth: {depth}")
        print(f"     Output: {y_labels}")
    
    # Analyze state distribution
    print("\nState distribution analysis:")
    empty_count = 0
    round_only_count = 0
    square_or_mixed_count = 0
    total_states = 0
    
    for _, y_labels in data:
        for label in y_labels:
            total_states += 1
            if label == [1, 0, 0]:  # empty
                empty_count += 1
            elif label == [0, 1, 0]:  # round only
                round_only_count += 1
            else:  # square or mixed
                square_or_mixed_count += 1
    
    print(f"  Total states: {total_states}")
    print(f"  Empty: {empty_count} ({empty_count/total_states:.3f})")
    print(f"  Round only: {round_only_count} ({round_only_count/total_states:.3f})")
    print(f"  Square or mixed: {square_or_mixed_count} ({square_or_mixed_count/total_states:.3f})")


def demonstrate_edge_cases():
    """Demonstrate edge cases and invalid Dyck-2 strings"""
    
    print("\n=== Edge Cases and Invalid Examples ===")
    
    dyck2_gen = Dyck2Language()
    
    # Edge cases
    edge_cases = [
        "",             # Empty string
        "()",           # Minimal valid
        "[]",           # Minimal valid
        "((()))",       # Deep round nesting
        "[[[]]]",       # Deep square nesting
        "()()()",       # Multiple concatenated
        "[][][]",       # Multiple concatenated
    ]
    
    print("Valid edge cases:")
    for case in edge_cases:
        is_valid = dyck2_gen.is_valid_dyck2(case)
        stack_states = dyck2_gen.output_generator(case)
        print(f"  '{case}' -> Valid: {is_valid}, States: {stack_states}")
    
    # Invalid cases
    invalid_cases = [
        "(",            # Unclosed round
        "[",            # Unclosed square
        ")",            # Unopened round
        "]",            # Unopened square
        "([)",          # Mismatched
        "[(])",         # Crossed brackets
        "([)]",         # Wrong nesting
        "(()",          # Unclosed
        "[]]",          # Unopened
    ]
    
    print("\nInvalid cases:")
    for case in invalid_cases:
        is_valid = dyck2_gen.is_valid_dyck2(case)
        stack_states = dyck2_gen.output_generator(case)
        print(f"  '{case}' -> Valid: {is_valid}, States: {stack_states}")


def demonstrate_parameter_variation():
    """Demonstrate how different parameters affect generation"""
    
    print("\n=== Parameter Variation Examples ===")
    
    # Different parameter combinations
    params = [
        (0.6, 0.2),  # High nesting, low concatenation
        (0.2, 0.6),  # Low nesting, high concatenation
        (0.4, 0.4),  # Balanced
        (0.8, 0.1),  # Very high nesting
        (0.1, 0.8),  # Very high concatenation
    ]
    
    for p, q in params:
        print(f"\nParameters: p={p}, q={q}")
        dyck2_gen = Dyck2Language(p=p, q=q)
        
        for i in range(3):
            string = dyck2_gen.generate(0, 12, 4)
            depth = dyck2_gen.get_depth(string)
            is_valid = dyck2_gen.is_valid_dyck2(string)
            print(f"  {i+1}. '{string}' -> Valid: {is_valid}, Depth: {depth}")


if __name__ == "__main__":
    print("Dyck-2 Language Usage Examples")
    print("=" * 50)
    
    demonstrate_dyck2_generation()
    demonstrate_stack_states()
    demonstrate_dataset_generation()
    demonstrate_edge_cases()
    demonstrate_parameter_variation()
    
    print("\n" + "=" * 50)
    print("Dyck-2 examples completed!") 