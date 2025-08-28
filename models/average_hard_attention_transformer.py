import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AverageAttentionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000)
        self.layers = nn.ModuleList([AverageAttentionBlock(d_model) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, 2)  # 输出2维，对应one-hot标签
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x, seq_len)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        logits = self.fc_out(x)  # (batch_size, seq_len, 2)
        return logits  # 直接输出logits，适配one-hot标签的交叉熵损失

class AverageAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Average attention across sequence
        avg = x.mean(dim=1, keepdim=True)  # (batch_size, 1, d_model)
        attn_output = self.linear(avg).expand_as(x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x, seq_len):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:seq_len, :].transpose(0, 1)

