"""
Feed-Forward Network implementation for transformer models.

This module contains the position-wise feed-forward network used in transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    A simple 2-layer fully connected feed-forward network with ReLU activation
    applied to each position separately and identically.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation + ReLU
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit Feed-Forward Network.
    
    An alternative to the standard FFN using Gated Linear Units (GLU).
    Often provides better performance for language models.
    
    GLU(x) = (xW1 + b1) ⊙ σ(xW2 + b2)W3 + b3
    where ⊙ is element-wise multiplication and σ is sigmoid.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the GLU feed-forward network.
        
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)  # Gate
        self.linear3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GLU feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Compute gate and value
        gate = torch.sigmoid(self.linear2(x))
        value = self.linear1(x)
        
        # Apply gating mechanism
        gated = gate * value
        gated = self.dropout(gated)
        
        # Final linear transformation
        output = self.linear3(gated)
        
        return output


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    Uses the SwiGLU activation function which has shown strong performance
    in recent language models like LLaMA and PaLM.
    
    SwiGLU(x) = Swish(xW1) ⊙ (xW2)
    where Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the SwiGLU feed-forward network.
        
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Note: We use d_ff * 2 for the hidden dimension to match the parameter count
        # of a standard FFN when using gating
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)
        self.linear3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Compute gate with Swish activation and value
        swish_gate = F.silu(self.linear1(x))  # SiLU is same as Swish
        value = self.linear2(x)
        
        # Apply gating mechanism
        gated = swish_gate * value
        gated = self.dropout(gated)
        
        # Final linear transformation
        output = self.linear3(gated)
        
        return output
