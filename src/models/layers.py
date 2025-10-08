"""
Transformer layer implementation.

This module contains the complete transformer encoder and decoder layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork


class TransformerEncoderLayer(nn.Module):
    """
    A single transformer encoder layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm (residual connection + layer normalization)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network hidden dimension
            dropout: Dropout probability
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)  # Add & Norm
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)  # Add & Norm
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    A single transformer decoder layer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (encoder-decoder attention)
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        """
        Initialize transformer decoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network hidden dimension
            dropout: Dropout probability
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        # Multi-head attention layers
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, tgt_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_len, d_model)
            self_attn_mask: Self-attention mask for target sequence
            cross_attn_mask: Cross-attention mask for source sequence
            
        Returns:
            Output tensor of shape (batch_size, tgt_len, d_model)
        """
        # Masked self-attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(x, x, x, self_attn_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attention(
            x, encoder_output, encoder_output, cross_attn_mask
        )
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)
        
        return x


class GPTDecoderLayer(nn.Module):
    """
    A GPT-style decoder layer (decoder-only architecture).
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    
    This is simpler than the full transformer decoder as it doesn't
    have cross-attention (no encoder).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        """
        Initialize GPT decoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network hidden dimension
            dropout: Dropout probability
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GPT decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional causal attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)
        
        return x
