#!/usr/bin/env python3
"""
Simplified Parity Language Training Script
For quick testing and validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

# 添加父目录到路径以导入模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.parity_dataset import get_parity_dataset
from models.softmax_attention_transformer import SoftmaxAttentionTransformer


def collate_fn(batch):
    """Data batch processing function"""
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


def calculate_accuracy(predictions, targets):
    """Calculate prediction accuracy"""
    pred_classes = torch.argmax(predictions, dim=-1)
    target_classes = torch.argmax(targets, dim=-1)
    correct = (pred_classes == target_classes).float()
    return correct.mean().item()


def train_simple_parity():
    """Simplified parity training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    print("Generating parity language dataset...")
    full_dataset = get_parity_dataset(
        n_samples=3000,  # smaller dataset for quick testing
        seq_len=8,       # shorter sequence length
        balanced=True
    )
    
    # Split dataset
    train_size = 1800
    val_size = 600
    test_size = 600
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Get vocabulary size
    vocab_size = len(full_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {full_dataset.vocab}")
    
    # Create model
    model = SoftmaxAttentionTransformer(
        vocab_size=vocab_size,
        d_model=32,  # smaller model
        n_heads=2,
        n_layers=1,
        dropout=0.0  # No dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("Starting training...")
    num_epochs = 10  # fewer epochs for quick testing
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(x_batch)
            
            # Calculate loss
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            accuracy = calculate_accuracy(predictions, y_batch)
            
            train_loss += loss.item()
            train_accuracy += accuracy
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                accuracy = calculate_accuracy(predictions, y_batch)
                
                val_loss += loss.item()
                val_accuracy += accuracy
                val_batches += 1
        
        # Calculate average loss and accuracy
        avg_train_loss = train_loss / train_batches
        avg_train_accuracy = train_accuracy / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_accuracy / val_batches
        
        # Record history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_accuracy)
        val_accuracies.append(avg_val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training loss: {avg_train_loss:.4f}, Training accuracy: {avg_train_accuracy:.4f}")
        print(f"  Validation loss: {avg_val_loss:.4f}, Validation accuracy: {avg_val_accuracy:.4f}")
    
    # Testing phase
    print("\nTesting model performance...")
    model.eval()
    test_accuracy = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(x_batch)
            accuracy = calculate_accuracy(predictions, y_batch)
            
            test_accuracy += accuracy
            test_batches += 1
    
    avg_test_accuracy = test_accuracy / test_batches
    print(f"Test accuracy: {avg_test_accuracy:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('parity_simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test model on examples
    print("\nTesting model on examples:")
    print("=" * 40)
    
    for i in range(3):
        x_str, y_labels = full_dataset.data[i]
        x_tensor = torch.tensor([[full_dataset.vocab[c] for c in x_str]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            predictions = model(x_tensor)
            pred_classes = torch.argmax(predictions, dim=-1)[0].cpu().numpy()
            target_classes = torch.argmax(torch.tensor(y_labels), dim=-1).numpy()
        
        print(f"Example {i+1}:")
        print(f"Input: {x_str}")
        print(f"True labels: {target_classes}")
        print(f"Predicted labels: {pred_classes}")
        print(f"Correct: {np.array_equal(pred_classes, target_classes)}")
        print()
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    train_simple_parity() 