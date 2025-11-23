#!/usr/bin/env python3
"""
Training Script for Nima - Applications Model.

Trains nima on application-specific data (canopy, swimTO, us-law-severity-map).
Uses the same infrastructure as train_technical.py.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from train_technical (reuse all functionality)
from train_technical import main as train_main, load_config, setup_training, train_with_monitoring, evaluate_on_test_set
import logging

logger = logging.getLogger(__name__)


def create_applications_config():
    """Create training configuration for applications model."""
    config = {
        'model': {
            'preset': 'tiny',  # Start small, can scale up
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 4,
            'd_ff': 1024,
            'max_seq_length': 512,
            'dropout': 0.1
        },
        'data': {
            'tokenizer_type': 'bpe',
            'tokenizer_path': 'data/processed/applications/tokenizer_bpe.json',
            'train_file': 'data/processed/applications/train.txt',
            'val_file': 'data/processed/applications/val.txt',
            'test_file': 'data/processed/applications/test.txt',
            'max_length': 512
        },
        'training': {
            'seed': 42,
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 0.0001,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'log_steps': 50,
            'eval_steps': 200,
            'save_steps': 200,
            'use_mixed_precision': False,
            'gradient_accumulation_steps': 2,
            'weight_decay': 0.01,
            'lr_scheduler': 'cosine',
            'checkpoint_dir': 'experiments/nima_applications',
            'early_stopping': {
                'enabled': True,
                'patience': 3,
                'min_delta': 0.001,
                'metric': 'val_loss',
                'mode': 'min'
            }
        },
        'monitoring': {
            'tensorboard': {
                'enabled': True,
                'log_dir': 'experiments/nima_applications/tensorboard'
            },
            'wandb': {
                'enabled': False
            },
            'plots': {
                'save_dir': 'experiments/nima_applications/plots'
            }
        },
        'evaluation': {
            'generation': {
                'enabled': True,
                'prompts': [
                    'What is Canopy?',
                    'Tell me about SwimTO',
                    'What does the US Law Severity Map show?',
                    'How does Canopy handle CSV imports?'
                ],
                'max_length': 150,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.95,
                'num_samples': 1
            }
        }
    }
    return config


def main():
    """Main training function for applications model."""
    import argparse
    import yaml
    import tempfile
    
    parser = argparse.ArgumentParser(description='Train Nima - Applications Model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training configuration file (optional, uses default if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation on test set')
    
    args = parser.parse_args()
    
    # Create or load config
    if args.config:
        config = load_config(args.config)
    else:
        config = create_applications_config()
        # Save config to temp file for train_technical compatibility
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            args.config = f.name
    
    logger.info("=" * 80)
    logger.info("TRAINING NIMA - APPLICATIONS MODEL")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    
    # Use train_technical's main logic
    # We'll call setup_training and train_with_monitoring directly
    import torch
    
    # Setup training components
    model, train_loader, val_loader, test_loader, tokenizer, training_config = setup_training(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")
    
    if args.eval_only:
        # Load checkpoint
        if args.resume:
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        if test_loader:
            evaluate_on_test_set(model, test_loader, tokenizer, config, device)
        else:
            logger.error("No test set specified in configuration!")
    else:
        # Train model
        results = train_with_monitoring(model, train_loader, val_loader, training_config, config)
        
        # Final evaluation on test set
        if test_loader:
            # Load best model
            best_checkpoint = Path(training_config.checkpoint_dir) / 'checkpoint_best.pt'
            if best_checkpoint.exists():
                logger.info(f"\nLoading best model from: {best_checkpoint}")
                checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            evaluate_on_test_set(model, test_loader, tokenizer, config, device)
        
        # Print final results
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Epoch: {results['best_epoch']}")
        logger.info(f"Best Val Loss: {results['best_val_loss']:.4f}")
        logger.info(f"Final Train Loss: {results['final_train_loss']:.4f}")
        logger.info(f"Final Val Loss: {results['final_val_loss']:.4f}")
        logger.info("=" * 80)
        logger.info(f"\nModel saved to: {training_config.checkpoint_dir}/checkpoint_best.pt")
        logger.info(f"Tokenizer saved to: {config['data']['tokenizer_path']}")


if __name__ == "__main__":
    main()





