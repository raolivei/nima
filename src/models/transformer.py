"""
Complete Transformer model implementations.

This module contains full transformer models including:
- Encoder-Decoder Transformer (original)
- GPT-style Decoder-only Transformer
- BERT-style Encoder-only Transformer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import PositionalEncoding
from .layers import TransformerEncoderLayer, TransformerDecoderLayer, GPTDecoderLayer


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal mask to prevent attention to future positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        Causal mask tensor of shape (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask to prevent attention to padding tokens.
    
    Args:
        seq: Input sequence tensor
        pad_idx: Padding token index
        
    Returns:
        Padding mask tensor
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


class TransformerModel(nn.Module):
    """
    Complete Transformer model (Encoder-Decoder architecture).
    
    This is the original transformer architecture from "Attention Is All You Need".
    Suitable for sequence-to-sequence tasks like translation.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        """
        Initialize the transformer model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            src_mask: Source mask tensor
            
        Returns:
            Encoded representation of shape (batch_size, src_len, d_model)
        """
        # Embedding + positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (src_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, src_len, d_model)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            encoder_output: Encoder output tensor
            tgt_mask: Target mask tensor
            src_mask: Source mask tensor
            
        Returns:
            Decoded representation of shape (batch_size, tgt_len, d_model)
        """
        # Embedding + positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (tgt_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, tgt_len, d_model)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Source sequence tensor
            tgt: Target sequence tensor
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor
            
        Returns:
            Output logits of shape (batch_size, tgt_len, tgt_vocab_size)
        """
        # Generate masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = create_causal_mask(tgt.size(1), tgt.device)
            tgt_padding_mask = create_padding_mask(tgt, self.pad_idx)
            tgt_mask = tgt_mask & tgt_padding_mask
        
        # Encode and decode
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return logits


class GPTModel(nn.Module):
    """
    GPT-style decoder-only transformer model.
    
    This is a decoder-only architecture suitable for autoregressive language modeling.
    Similar to GPT, GPT-2, GPT-3, etc.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        """
        Initialize the GPT model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization and output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (common practice in language models)
        self.output_projection.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the GPT model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask tensor
            
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len, input_ids.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & attention_mask
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(generated)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit pad token
                if next_token.item() == pad_token_id:
                    break
        
        return generated
