import torch
import torch.nn as nn
import numpy as np


class TransformerLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=8, d_ff=3072, dropout=0.1):
        super().__init__()
        # Multi-Head Self-Attention
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)  # For Q, K, V
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward Neural Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 1: Multi-Head Self-Attention
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # Shape: [batch_size, seq_len, 3 * d_model]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split into Q, K, V
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)  # Shape: [batch_size, num_heads, seq_len, seq_len]
        attn_output = torch.matmul(attn, v)  # Shape: [batch_size, num_heads, seq_len, head_dim]

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output)

        # Residual Connection + LayerNorm
        x = self.norm1(x + self.dropout(attn_output))

        # Step 2: Feedforward Neural Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

if __name__ == "__main__" :
    # Example Usage
    batch_size, seq_len, d_model = 2, 8, 768
    x = torch.rand(batch_size, seq_len, d_model)  # Random input embeddings
    transformer_layer = TransformerLayer()
    
    output = transformer_layer(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("\nx subset: ", x[:20])
    print("\noutput subset: ", output[:20])
    