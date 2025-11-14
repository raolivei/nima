"""
Inference engine and text generation utilities.

This package provides text generation capabilities:
- Multiple sampling strategies (greedy, top-k, top-p, beam search)
- Temperature and repetition control
- Batch generation
- Interactive conversation mode
"""

from .generator import (
    TextGenerator,
    InteractiveGenerator,
    batch_generate
)

__all__ = [
    'TextGenerator',
    'InteractiveGenerator',
    'batch_generate'
]
