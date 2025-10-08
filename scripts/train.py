#!/usr/bin/env python3
"""
Training script for Nima LLM.

This script demonstrates how to train a language model using the training framework.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    from models import create_model_from_preset, print_model_info
    from data import (
        prepare_dataset,
        create_dataloader,
        DataCollator,
        CausalLMDataset
    )
    from training import (
        Trainer,
        TrainingConfig,
        MetricsTracker,
        EarlyStopping
    )
    from utils import Config
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Training modules not available: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    TRAINING_AVAILABLE = False


def train_from_config(config_path: str):
    """
    Train a model from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    print("=" * 80)
    print("Nima LLM Training")
    print("=" * 80)
    
    # Load configuration
    config = Config.load_config(config_path)
    
    print(f"\nLoading configuration from: {config_path}")
    print(f"Model: {config.model.d_model}d, {config.model.n_layers} layers")
    print(f"Training: {config.training.num_epochs} epochs, batch size {config.training.batch_size}")
    
    # Prepare dataset
    print("\n" + "-" * 80)
    print("Preparing dataset...")
    print("-" * 80)
    
    train_dataset, val_dataset, tokenizer = prepare_dataset(
        dataset_name=config.data.dataset,
        tokenizer_type=config.data.tokenizer_type,
        data_dir=config.paths.data_dir,
        max_length=config.model.max_seq_len,
        vocab_size=config.model.vocab_size,
        train_split=config.data.train_split
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data collator and loaders
    collator = DataCollator(tokenizer)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating model...")
    print("-" * 80)
    
    model = create_model_from_preset(
        'gpt-tiny',  # Start with tiny model
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    
    print_model_info(model, "Nima GPT")
    
    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        gradient_clip=config.training.gradient_clip,
        save_steps=config.training.save_every,
        eval_steps=config.training.eval_every,
        checkpoint_dir=config.paths.checkpoint_dir,
        log_dir=config.paths.log_dir
    )
    
    # Create trainer
    print("\n" + "-" * 80)
    print("Initializing trainer...")
    print("-" * 80)
    
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader
    )
    
    # Train
    print("\n" + "-" * 80)
    print("Starting training...")
    print("-" * 80)
    
    results = trainer.train()
    
    # Print results
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Final train loss: {results['final_train_loss']:.4f}")
    if results['final_eval_loss'] is not None:
        print(f"Final eval loss: {results['final_eval_loss']:.4f}")
    
    return results


def train_quick_test():
    """Quick training test with tiny model and dataset."""
    print("=" * 80)
    print("Quick Training Test")
    print("=" * 80)
    
    # Prepare small dataset
    print("\nPreparing tiny shakespeare dataset...")
    train_dataset, val_dataset, tokenizer = prepare_dataset(
        dataset_name='tiny_shakespeare',
        tokenizer_type='char',
        data_dir='data',
        max_length=128,  # Short sequences for quick test
        train_split=0.9
    )
    
    # Create model
    print("\nCreating tiny model...")
    model = create_model_from_preset(
        'gpt-tiny',
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128
    )
    
    print_model_info(model, "Test Model")
    
    # Create data loaders
    collator = DataCollator(tokenizer)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator
    )
    
    # Training config for quick test
    config = TrainingConfig(
        num_epochs=2,
        learning_rate=3e-4,
        batch_size=4,
        warmup_steps=100,
        save_steps=500,
        eval_steps=250,
        logging_steps=50
    )
    
    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader
    )
    
    print("\nStarting quick test training...")
    results = trainer.train()
    
    print("\nQuick test completed!")
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Nima LLM')
    parser.add_argument('--config', type=str, default='configs/base_model.yaml',
                       help='Path to config file')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick training test')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if not TRAINING_AVAILABLE:
        print("Error: Training dependencies not available.")
        print("Please run: pip install -r requirements.txt")
        return
    
    try:
        if args.quick_test:
            # Run quick test
            results = train_quick_test()
        else:
            # Train from config
            results = train_from_config(args.config)
            
        print("\nTraining script completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()