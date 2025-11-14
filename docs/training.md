# Nima Training Guide

This guide explains how to train your own language models using the Nima framework.

## Quick Start

### 1. Prepare Data

First, prepare your training data:

```bash
python scripts/prepare_data.py --dataset tiny_shakespeare --tokenizer char
```

### 2. Train a Model

Train with default configuration:

```bash
python scripts/train.py --config configs/base_model.yaml
```

Or run a quick test:

```bash
python scripts/train.py --quick_test
```

## Training Configuration

Edit `configs/base_model.yaml` to customize training:

```yaml
model:
  vocab_size: 10000
  d_model: 512 # Model dimension
  n_heads: 8 # Number of attention heads
  n_layers: 6 # Number of transformer layers
  d_ff: 2048 # Feed-forward dimension
  max_seq_len: 1024 # Maximum sequence length
  dropout: 0.1

training:
  num_epochs: 100
  learning_rate: 0.0001
  batch_size: 32
  warmup_steps: 4000
  weight_decay: 0.01
  gradient_clip: 1.0
  save_every: 1000 # Save checkpoint every N steps
  eval_every: 500 # Evaluate every N steps
```

## Advanced Training

### Resume from Checkpoint

```bash
python scripts/train.py --config configs/base_model.yaml --checkpoint experiments/checkpoints/step_5000.pt
```

### Custom Training Loop

```python
from models import create_gpt_small
from data import prepare_dataset, create_dataloader, DataCollator
from training import Trainer, TrainingConfig

# Prepare data
train_dataset, val_dataset, tokenizer = prepare_dataset(
    dataset_name='tiny_shakespeare',
    tokenizer_type='char',
    max_length=512
)

# Create model
model = create_gpt_small(vocab_size=tokenizer.vocab_size)

# Setup data loaders
collator = DataCollator(tokenizer)
train_loader = create_dataloader(train_dataset, batch_size=32, collate_fn=collator)
val_loader = create_dataloader(val_dataset, batch_size=32, collate_fn=collator)

# Configure training
config = TrainingConfig(
    num_epochs=10,
    learning_rate=3e-4,
    warmup_steps=1000,
    save_steps=1000,
    eval_steps=500
)

# Train
trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=val_loader
)

results = trainer.train()
```

## Training Features

### Mixed Precision Training

Enabled by default for faster training on GPUs:

```python
config = TrainingConfig(
    use_mixed_precision=True  # Uses automatic mixed precision (AMP)
)
```

### Gradient Accumulation

Train with effectively larger batch sizes:

```python
config = TrainingConfig(
    batch_size=16,
    gradient_accumulation_steps=4  # Effective batch size: 16 * 4 = 64
)
```

### Learning Rate Scheduling

Multiple scheduler types available:

```python
config = TrainingConfig(
    lr_scheduler_type='cosine',  # 'cosine', 'linear', 'constant'
    warmup_steps=1000,
    min_lr=0.0
)
```

### Early Stopping

Prevent overfitting with early stopping:

```python
from training import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,      # Stop after 10 epochs without improvement
    min_delta=0.001   # Minimum improvement threshold
)

# In training loop
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = evaluate()

    if early_stopping(val_loss):
        print("Early stopping triggered!")
        break
```

## Monitoring Training

### Metrics Tracking

```python
from training import MetricsTracker

tracker = MetricsTracker(log_dir='experiments/logs')

# Log metrics during training
tracker.log({'loss': loss, 'lr': learning_rate}, step=global_step)

# Get summary
summary = tracker.get_summary()
print(f"Average loss: {summary['loss']['mean']}")

# Save metrics
tracker.save('training_metrics.json')
```

### Progress Display

```python
from training import ProgressPrinter

printer = ProgressPrinter(
    total_steps=10000,
    metrics_to_track=['loss', 'lr', 'perplexity'],
    print_frequency=10
)

# Print step progress
printer.print_step(step, {'loss': loss, 'lr': lr})

# Print epoch summary
printer.print_epoch_end(epoch, {'train_loss': train_loss, 'val_loss': val_loss})
```

## Optimization

### Optimizers

Choose from multiple optimizers:

```python
from training import get_optimizer

# AdamW (recommended)
optimizer = get_optimizer(
    model,
    optimizer_type='adamw',
    learning_rate=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Adam
optimizer = get_optimizer(model, optimizer_type='adam')

# SGD with momentum
optimizer = get_optimizer(model, optimizer_type='sgd', momentum=0.9)

# AdamWScale (for large models)
optimizer = get_optimizer(model, optimizer_type='adamw_scale')
```

### Learning Rate Schedulers

```python
from training import get_scheduler

# Cosine annealing with warmup
scheduler = get_scheduler(
    optimizer,
    scheduler_type='cosine',
    num_training_steps=10000,
    num_warmup_steps=1000,
    min_lr=0.0
)

# Cosine with restarts
scheduler = get_scheduler(
    optimizer,
    scheduler_type='cosine_restarts',
    num_training_steps=10000,
    num_warmup_steps=1000
)
```

## Checkpointing

### Automatic Checkpointing

Checkpoints are saved automatically during training:

- `step_N.pt`: Regular checkpoints at specified intervals
- `epoch_N.pt`: End-of-epoch checkpoints
- `best_model.pt`: Best model based on validation loss

### Manual Checkpointing

```python
# Save checkpoint
trainer.save_checkpoint('my_checkpoint')

# Load checkpoint
trainer.load_checkpoint('experiments/checkpoints/my_checkpoint.pt')
```

### Checkpoint Contents

Each checkpoint contains:

- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training step and epoch
- Best validation loss
- Mixed precision scaler state (if used)

## Performance Tips

### GPU Training

```python
config = TrainingConfig(
    device='cuda',  # Use GPU
    use_mixed_precision=True  # Faster training
)
```

### Batch Size Tuning

Find optimal batch size for your GPU:

```python
# Start small and increase
batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    try:
        config = TrainingConfig(batch_size=batch_size)
        trainer = Trainer(model, config, train_loader)
        # Test one batch
        break
    except RuntimeError:  # Out of memory
        continue
```

### Gradient Accumulation

Simulate larger batches with less memory:

```python
config = TrainingConfig(
    batch_size=8,
    gradient_accumulation_steps=8  # Effective batch size: 64
)
```

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Enable gradient accumulation
3. Reduce model size (d_model, n_layers)
4. Use gradient checkpointing (coming soon)

### Training Instability

1. Reduce learning rate
2. Increase warmup steps
3. Enable gradient clipping
4. Check for NaN values in data

### Slow Training

1. Enable mixed precision training
2. Increase batch size
3. Use multiple GPUs (coming soon)
4. Optimize data loading (num_workers)

## Next Steps

- [Evaluation Guide](evaluation.md) - Evaluate trained models
- [Inference Guide](inference.md) - Generate text with trained models
- [Model Architecture](architecture.md) - Understand the transformer implementation

## Examples

See `notebooks/` for training examples:

- `training_basics.ipynb` - Basic training workflow
- `advanced_training.ipynb` - Advanced techniques
- `hyperparameter_tuning.ipynb` - Finding optimal hyperparameters
