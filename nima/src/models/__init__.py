"""
Model architectures and components.

This package contains all the building blocks for transformer-based language models:
- Attention mechanisms (scaled dot-product, multi-head)
- Feed-forward networks (standard, GLU, SwiGLU)
- Transformer layers (encoder, decoder, GPT-style)
- Complete model architectures (Transformer, GPT)
"""

# Core attention mechanisms
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding
)

# Feed-forward networks
from .feedforward import (
    FeedForwardNetwork,
    GLUFeedForward,
    SwiGLUFeedForward
)

# Transformer layers
from .layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    GPTDecoderLayer
)

# Complete models
from .transformer import (
    TransformerModel,
    GPTModel,
    create_causal_mask,
    create_padding_mask
)

# Model factory functions
from .factory import (
    create_gpt_small,
    create_gpt_medium,
    create_gpt_large,
    create_transformer_base,
    create_transformer_big,
    create_custom_model,
    create_model_from_preset,
    get_model_size,
    print_model_info,
    MODEL_PRESETS
)

__all__ = [
    # Attention
    'ScaledDotProductAttention',
    'MultiHeadAttention', 
    'PositionalEncoding',
    # Feed-forward
    'FeedForwardNetwork',
    'GLUFeedForward',
    'SwiGLUFeedForward',
    # Layers
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'GPTDecoderLayer',
    # Models
    'TransformerModel',
    'GPTModel',
    'create_causal_mask',
    'create_padding_mask',
    # Factory functions
    'create_gpt_small',
    'create_gpt_medium',
    'create_gpt_large',
    'create_transformer_base',
    'create_transformer_big',
    'create_custom_model',
    'create_model_from_preset',
    'get_model_size',
    'print_model_info',
    'MODEL_PRESETS'
]
