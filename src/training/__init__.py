"""
Training loops and optimization utilities.

This package provides comprehensive training capabilities:
- Trainer class with training loops
- Optimizers and learning rate schedulers
- Metrics tracking and monitoring
- Checkpointing and model saving
"""

# Core trainer
from .trainer import Trainer, TrainingConfig

# Optimization utilities
from .optimization import (
    get_optimizer,
    get_scheduler,
    AdamWScale,
    LinearWarmupCosineAnnealingLR,
    CosineAnnealingWarmupRestarts
)

# Metrics and monitoring
from .metrics import (
    MetricsTracker,
    ProgressPrinter,
    EarlyStopping,
    calculate_perplexity,
    calculate_tokens_per_second,
    estimate_training_time
)

__all__ = [
    # Trainer
    'Trainer',
    'TrainingConfig',
    # Optimization
    'get_optimizer',
    'get_scheduler',
    'AdamWScale',
    'LinearWarmupCosineAnnealingLR',
    'CosineAnnealingWarmupRestarts',
    # Metrics
    'MetricsTracker',
    'ProgressPrinter',
    'EarlyStopping',
    'calculate_perplexity',
    'calculate_tokens_per_second',
    'estimate_training_time'
]
