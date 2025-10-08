"""
Evaluation metrics and assessment tools.

This package provides comprehensive evaluation capabilities:
- Perplexity calculation
- Token accuracy metrics
- BLEU score
- Model comparison tools
"""

from .metrics import (
    calculate_perplexity,
    calculate_accuracy,
    calculate_top_k_accuracy,
    calculate_bleu_score,
    EvaluationMetrics,
    evaluate_model,
    compare_models,
    print_evaluation_report
)

__all__ = [
    'calculate_perplexity',
    'calculate_accuracy',
    'calculate_top_k_accuracy',
    'calculate_bleu_score',
    'EvaluationMetrics',
    'evaluate_model',
    'compare_models',
    'print_evaluation_report'
]
