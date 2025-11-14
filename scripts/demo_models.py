#!/usr/bin/env python3
"""
Example script demonstrating how to create and use transformer models.

This script shows:
1. How to create different model architectures
2. Basic model usage and forward passes
3. Model size and parameter information
"""

import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    from models import (
        create_gpt_small, 
        create_model_from_preset,
        print_model_info,
        GPTModel
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch not available: {e}")
    print("Please install PyTorch to run this example.")
    TORCH_AVAILABLE = False


def demo_gpt_model():
    """Demonstrate GPT model creation and usage."""
    print("=" * 60)
    print("GPT Model Demo")
    print("=" * 60)
    
    # Create a small GPT model
    vocab_size = 10000
    model = create_gpt_small(vocab_size)
    
    # Print model information
    print_model_info(model, "GPT Small")
    
    # Create some dummy input
    batch_size = 2
    seq_len = 50
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")
    
    # Demonstrate text generation
    print("\nDemonstrating text generation...")
    prompt = torch.randint(0, vocab_size, (1, 10))  # Random prompt
    
    with torch.no_grad():
        generated = model.generate(
            prompt, 
            max_length=20, 
            temperature=0.8,
            top_k=50
        )
    
    print(f"Generated sequence length: {generated.shape[1]}")
    print(f"Generated tokens: {generated[0].tolist()}")


def demo_model_presets():
    """Demonstrate model creation from presets."""
    print("\n" + "=" * 60)
    print("Model Presets Demo")
    print("=" * 60)
    
    vocab_size = 5000
    
    # Available presets
    presets = ['gpt-tiny', 'gpt-small']
    
    for preset in presets:
        print(f"\nCreating model with preset: {preset}")
        
        model = create_model_from_preset(preset, vocab_size)
        print_model_info(model, f"Model ({preset})")


def demo_custom_model():
    """Demonstrate custom model configuration."""
    print("\n" + "=" * 60)
    print("Custom Model Demo")
    print("=" * 60)
    
    # Create a very small model for demonstration
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1
    }
    
    model = GPTModel(**config)
    print_model_info(model, "Custom Tiny Model")
    
    # Test forward pass
    input_ids = torch.randint(0, config['vocab_size'], (1, 20))
    
    with torch.no_grad():
        logits = model(input_ids)
        print(f"\nForward pass successful!")
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")


def demo_model_comparison():
    """Compare different model sizes."""
    print("\n" + "=" * 60)
    print("Model Size Comparison")
    print("=" * 60)
    
    vocab_size = 10000
    models = {
        'Tiny': create_model_from_preset('gpt-tiny', vocab_size),
        'Small': create_model_from_preset('gpt-small', vocab_size),
    }
    
    print(f"{'Model':<10} {'Parameters':<15} {'Size (MB)':<12}")
    print("-" * 40)
    
    for name, model in models.items():
        from models.factory import get_model_size
        info = get_model_size(model)
        print(f"{name:<10} {info['total_parameters']:<15,} {info['model_size_mb']:<12.2f}")


def main():
    """Run all demonstrations."""
    if not TORCH_AVAILABLE:
        return
    
    print("Transformer Model Architecture Demo")
    print("This script demonstrates the transformer models we've built from scratch.")
    
    try:
        # Run demonstrations
        demo_gpt_model()
        demo_model_presets()
        demo_custom_model()
        demo_model_comparison()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
