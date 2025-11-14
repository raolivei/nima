"""
Model factory for creating pre-configured transformer models.

This module provides convenient functions to create models with common configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .transformer import GPTModel, TransformerModel


def create_gpt_small(vocab_size: int, **kwargs) -> GPTModel:
    """
    Create a small GPT model (similar to GPT-1 small).
    
    Args:
        vocab_size: Vocabulary size
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Small GPT model
    """
    config = {
        'vocab_size': vocab_size,
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    config.update(kwargs)
    return GPTModel(**config)


def create_gpt_medium(vocab_size: int, **kwargs) -> GPTModel:
    """
    Create a medium GPT model (similar to GPT-2 medium).
    
    Args:
        vocab_size: Vocabulary size
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Medium GPT model
    """
    config = {
        'vocab_size': vocab_size,
        'd_model': 1024,
        'n_heads': 16,
        'n_layers': 24,
        'd_ff': 4096,
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    config.update(kwargs)
    return GPTModel(**config)


def create_gpt_large(vocab_size: int, **kwargs) -> GPTModel:
    """
    Create a large GPT model (similar to GPT-2 large).
    
    Args:
        vocab_size: Vocabulary size
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Large GPT model
    """
    config = {
        'vocab_size': vocab_size,
        'd_model': 1280,
        'n_heads': 20,
        'n_layers': 36,
        'd_ff': 5120,
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    config.update(kwargs)
    return GPTModel(**config)


def create_transformer_base(src_vocab_size: int, tgt_vocab_size: int, **kwargs) -> TransformerModel:
    """
    Create a base transformer model (similar to the original paper).
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Base transformer model
    """
    config = {
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 5000,
        'dropout': 0.1
    }
    config.update(kwargs)
    return TransformerModel(**config)


def create_transformer_big(src_vocab_size: int, tgt_vocab_size: int, **kwargs) -> TransformerModel:
    """
    Create a big transformer model (similar to the original paper's big model).
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Big transformer model
    """
    config = {
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': 1024,
        'n_heads': 16,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 4096,
        'max_seq_len': 5000,
        'dropout': 0.3
    }
    config.update(kwargs)
    return TransformerModel(**config)


def create_custom_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Create a custom model from configuration.
    
    Args:
        model_type: Type of model ('gpt', 'transformer')
        config: Model configuration dictionary
        
    Returns:
        Configured model
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == 'gpt':
        return GPTModel(**config)
    elif model_type.lower() == 'transformer':
        return TransformerModel(**config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print detailed model information.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    info = get_model_size(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Non-trainable Parameters: {info['non_trainable_parameters']:,}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB")
    print(f"  Model Type: {type(model).__name__}")
    
    # Print layer information for GPT models
    if hasattr(model, 'layers'):
        print(f"  Number of Layers: {len(model.layers)}")
    if hasattr(model, 'd_model'):
        print(f"  Model Dimension: {model.d_model}")


# Pre-configured model presets
MODEL_PRESETS = {
    'gpt-tiny': {
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 1024,
        'max_seq_len': 512,
        'dropout': 0.1
    },
    'gpt-small': {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'max_seq_len': 1024,
        'dropout': 0.1
    },
    'gpt-medium': {
        'd_model': 1024,
        'n_heads': 16,
        'n_layers': 24,
        'd_ff': 4096,
        'max_seq_len': 1024,
        'dropout': 0.1
    },
    'transformer-base': {
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 5000,
        'dropout': 0.1
    }
}


def create_model_from_preset(preset_name: str, vocab_size: int, **kwargs) -> nn.Module:
    """
    Create a model from a preset configuration.
    
    Args:
        preset_name: Name of the preset configuration
        vocab_size: Vocabulary size
        **kwargs: Additional arguments to override preset defaults
        
    Returns:
        Configured model
        
    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in MODEL_PRESETS:
        available_presets = list(MODEL_PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available_presets}")
    
    config = MODEL_PRESETS[preset_name].copy()
    config.update(kwargs)
    config['vocab_size'] = vocab_size
    
    if preset_name.startswith('gpt'):
        return GPTModel(**config)
    elif preset_name.startswith('transformer'):
        # For transformer models, assume same vocab size for src and tgt
        config['src_vocab_size'] = vocab_size
        config['tgt_vocab_size'] = vocab_size
        return TransformerModel(**config)
    else:
        raise ValueError(f"Unknown preset type for '{preset_name}'")
