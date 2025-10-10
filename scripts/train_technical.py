#!/usr/bin/env python3
"""
Enhanced Training Script for Nima - Technical Model.

Features:
- Multi-source data loading
- Early stopping
- Learning rate scheduling
- TensorBoard and W&B logging
- Automatic checkpointing
- Metrics visualization
- Test set evaluation
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch
import logging
from typing import Dict, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model_from_preset, GPTModel
from data import create_tokenizer, TextDataset, create_dataloader, DataCollator
from training import Trainer, TrainingConfig
from training.monitoring import (
    EarlyStopping, MetricsTracker, TensorBoardLogger, 
    WandBLogger, compute_metrics
)
from inference import TextGenerator
from evaluation import evaluate_model, print_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_training(config: Dict) -> tuple:
    """
    Set up all training components.
    
    Returns:
        (model, train_loader, val_loader, test_loader, tokenizer, training_config)
    """
    logger.info("=" * 80)
    logger.info("SETTING UP TRAINING FOR NIMA - TECHNICAL MODEL")
    logger.info("=" * 80)
    
    # Set random seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load tokenizer
    tokenizer_path = config['data']['tokenizer_path']
    tokenizer_type = config['data']['tokenizer_type']
    
    logger.info(f"Loading {tokenizer_type} tokenizer from: {tokenizer_path}")
    tokenizer = create_tokenizer(tokenizer_type)
    tokenizer.load(tokenizer_path)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model_config = config['model']
    if 'preset' in model_config and model_config['preset']:
        logger.info(f"Creating model from preset: {model_config['preset']}")
        model = create_model_from_preset(
            model_config['preset'],
            vocab_size=tokenizer.vocab_size
        )
    else:
        logger.info("Creating custom model")
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            d_model=model_config.get('d_model', 768),
            n_layers=model_config.get('n_layers', 12),
            n_heads=model_config.get('n_heads', 12),
            d_ff=model_config.get('d_ff', 3072),
            max_seq_length=model_config.get('max_seq_length', 512),
            dropout=model_config.get('dropout', 0.1)
        )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Load datasets
    data_config = config['data']
    max_length = data_config.get('max_length', 512)
    
    logger.info("\nLoading datasets...")
    
    # Training data
    with open(data_config['train_file'], 'r', encoding='utf-8') as f:
        train_text = f.read()
    train_dataset = TextDataset([train_text], tokenizer, max_length=max_length)
    logger.info(f"Training samples: {len(train_dataset)}")
    
    # Validation data
    with open(data_config['val_file'], 'r', encoding='utf-8') as f:
        val_text = f.read()
    val_dataset = TextDataset([val_text], tokenizer, max_length=max_length)
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Test data
    test_dataset = None
    if 'test_file' in data_config and data_config['test_file']:
        with open(data_config['test_file'], 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = TextDataset([test_text], tokenizer, max_length=max_length)
        logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_config = config['training']
    collator = DataCollator(tokenizer)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=train_config.get('shuffle', True),
        collate_fn=collator,
        num_workers=train_config.get('dataloader_num_workers', 0)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config.get('eval_batch_size', train_config['batch_size']),
        shuffle=False,
        collate_fn=collator
    )
    
    test_loader = None
    if test_dataset:
        test_loader = create_dataloader(
            test_dataset,
            batch_size=train_config.get('eval_batch_size', train_config['batch_size']),
            shuffle=False,
            collate_fn=collator
        )
    
    # Create training config
    training_config = TrainingConfig(
        output_dir=train_config['checkpoint_dir'],
        num_epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        learning_rate=train_config['learning_rate'],
        warmup_steps=train_config.get('warmup_steps', 0),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        log_interval=train_config.get('log_steps', 50),
        eval_interval=train_config.get('eval_steps', 500),
        save_interval=train_config.get('save_steps', 500),
        use_mixed_precision=train_config.get('use_mixed_precision', False)
    )
    
    return model, train_loader, val_loader, test_loader, tokenizer, training_config


def train_with_monitoring(
    model,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    config: Dict
) -> Dict:
    """
    Train model with comprehensive monitoring.
    
    Returns:
        Dictionary with training results
    """
    # Setup monitoring
    train_config = config['training']
    monitoring_config = config.get('monitoring', {})
    
    # Early stopping
    early_stopping = None
    if train_config.get('early_stopping', {}).get('enabled', False):
        es_config = train_config['early_stopping']
        early_stopping = EarlyStopping(
            patience=es_config.get('patience', 5),
            min_delta=es_config.get('min_delta', 0.001),
            metric=es_config.get('metric', 'val_loss'),
            mode=es_config.get('mode', 'min')
        )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(
        metrics=['train_loss', 'val_loss', 'train_perplexity', 'val_perplexity', 
                'learning_rate', 'grad_norm'],
        save_dir=monitoring_config.get('plots', {}).get('save_dir')
    )
    
    # TensorBoard
    tb_logger = None
    if monitoring_config.get('tensorboard', {}).get('enabled', False):
        tb_logger = TensorBoardLogger(
            log_dir=monitoring_config['tensorboard']['log_dir'],
            enabled=True
        )
    
    # Weights & Biases
    wandb_logger = None
    if monitoring_config.get('wandb', {}).get('enabled', False):
        wandb_config = monitoring_config['wandb']
        wandb_logger = WandBLogger(
            project=wandb_config['project'],
            name=wandb_config['name'],
            config=config,
            enabled=True,
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', '')
        )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    results = {
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'final_train_loss': 0,
        'final_val_loss': 0
    }
    
    try:
        for epoch in range(training_config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
            logger.info("-" * 80)
            
            # Training epoch
            train_loss = trainer.train_epoch()
            train_perplexity = torch.exp(torch.tensor(train_loss)).item()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
            
            # Validation
            val_metrics = evaluate_model(model, val_loader, trainer.device)
            val_loss = val_metrics['loss']
            val_perplexity = val_metrics['perplexity']
            
            logger.info(f"Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            
            # Track metrics
            step = trainer.global_step
            metrics_tracker.add(
                step=step,
                train_loss=train_loss,
                val_loss=val_loss,
                train_perplexity=train_perplexity,
                val_perplexity=val_perplexity,
                learning_rate=trainer.get_lr()
            )
            
            # Log to TensorBoard
            if tb_logger:
                tb_logger.log_scalar('train/loss', train_loss, step)
                tb_logger.log_scalar('val/loss', val_loss, step)
                tb_logger.log_scalar('train/perplexity', train_perplexity, step)
                tb_logger.log_scalar('val/perplexity', val_perplexity, step)
                tb_logger.log_scalar('learning_rate', trainer.get_lr(), step)
            
            # Log to W&B
            if wandb_logger:
                wandb_logger.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'train/perplexity': train_perplexity,
                    'val/perplexity': val_perplexity,
                    'learning_rate': trainer.get_lr()
                }, step=step)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                results['best_epoch'] = epoch + 1
                results['best_val_loss'] = val_loss
                
                checkpoint_path = Path(training_config.output_dir) / 'checkpoint_best.pt'
                trainer.save_checkpoint(str(checkpoint_path), val_loss)
                logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping check
            if early_stopping:
                if early_stopping(val_loss, epoch + 1):
                    logger.info("Early stopping triggered!")
                    break
            
            # Plot metrics periodically
            if (epoch + 1) % 5 == 0 and metrics_tracker.save_dir:
                plot_path = metrics_tracker.save_dir / f'metrics_epoch_{epoch+1}.png'
                metrics_tracker.plot_loss_curves(str(plot_path))
        
        results['final_train_loss'] = train_loss
        results['final_val_loss'] = val_loss
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user!")
    
    finally:
        # Final plots
        if metrics_tracker.save_dir:
            final_plot = metrics_tracker.save_dir / 'final_metrics.png'
            metrics_tracker.plot_loss_curves(str(final_plot))
            logger.info(f"Saved final metrics plot to: {final_plot}")
        
        # Print summary
        metrics_tracker.print_summary()
        
        # Close loggers
        if tb_logger:
            tb_logger.close()
        if wandb_logger:
            wandb_logger.finish()
    
    return results


def evaluate_on_test_set(
    model,
    test_loader,
    tokenizer,
    config: Dict,
    device: torch.device
):
    """Evaluate model on test set with sample generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    
    # Compute metrics
    test_metrics = evaluate_model(model, test_loader, device)
    print_evaluation_report(test_metrics, "Test Set Results")
    
    # Generate samples
    eval_config = config.get('evaluation', {})
    if eval_config.get('generation', {}).get('enabled', False):
        gen_config = eval_config['generation']
        prompts = gen_config.get('prompts', [])
        
        if prompts:
            logger.info("\n" + "=" * 80)
            logger.info("SAMPLE GENERATIONS")
            logger.info("=" * 80)
            
            generator = TextGenerator(model, tokenizer, device)
            
            for prompt in prompts:
                logger.info(f"\nPrompt: \"{prompt}\"")
                logger.info("-" * 80)
                
                texts = generator.generate(
                    prompt=prompt,
                    max_length=gen_config.get('max_length', 100),
                    temperature=gen_config.get('temperature', 0.8),
                    top_k=gen_config.get('top_k', 50),
                    top_p=gen_config.get('top_p', 0.95),
                    num_return_sequences=gen_config.get('num_samples', 1)
                )
                
                for i, text in enumerate(texts, 1):
                    print(f"\nSample {i}:")
                    print(text)
                    print()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Nima - Technical Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation on test set')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
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
            checkpoint = torch.load(args.resume, map_location=device)
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
            best_checkpoint = Path(training_config.output_dir) / 'checkpoint_best.pt'
            if best_checkpoint.exists():
                logger.info(f"\nLoading best model from: {best_checkpoint}")
                checkpoint = torch.load(best_checkpoint, map_location=device)
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


if __name__ == "__main__":
    main()
