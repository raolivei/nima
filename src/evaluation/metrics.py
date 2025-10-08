"""
Evaluation metrics for language models.

This module provides comprehensive metrics for evaluating LLM performance including:
- Perplexity
- BLEU score
- Token accuracy
- Loss metrics
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return math.exp(loss)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        predictions: Predicted token IDs (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Accuracy as a percentage
    """
    # Get predicted token IDs
    pred_ids = predictions.argmax(dim=-1)
    
    # Create mask for valid tokens
    mask = targets != ignore_index
    
    # Calculate accuracy
    correct = (pred_ids == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item() * 100


def calculate_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Predicted logits (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        k: Number of top predictions to consider
        ignore_index: Index to ignore
        
    Returns:
        Top-k accuracy as a percentage
    """
    # Get top-k predictions
    top_k_preds = predictions.topk(k, dim=-1).indices
    
    # Expand targets to match top-k shape
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    
    # Create mask for valid tokens
    mask = targets != ignore_index
    
    # Check if target is in top-k
    correct = (top_k_preds == targets_expanded).any(dim=-1) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item() * 100


def calculate_bleu_score(
    generated_texts: List[str],
    reference_texts: List[str],
    n_gram: int = 4
) -> float:
    """
    Calculate BLEU score for generated text.
    
    A simplified BLEU implementation for evaluation.
    
    Args:
        generated_texts: List of generated text strings
        reference_texts: List of reference text strings
        n_gram: Maximum n-gram size (default: 4)
        
    Returns:
        BLEU score (0-100)
    """
    def get_ngrams(text: str, n: int) -> Counter:
        """Extract n-grams from text."""
        words = text.split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    if len(generated_texts) != len(reference_texts):
        raise ValueError("Number of generated and reference texts must match")
    
    total_scores = []
    
    for gen_text, ref_text in zip(generated_texts, reference_texts):
        scores = []
        
        for n in range(1, n_gram + 1):
            gen_ngrams = get_ngrams(gen_text, n)
            ref_ngrams = get_ngrams(ref_text, n)
            
            if len(gen_ngrams) == 0 or len(ref_ngrams) == 0:
                scores.append(0.0)
                continue
            
            # Calculate precision
            matches = sum((gen_ngrams & ref_ngrams).values())
            total = sum(gen_ngrams.values())
            
            precision = matches / total if total > 0 else 0.0
            scores.append(precision)
        
        # Geometric mean of precisions
        if all(s > 0 for s in scores):
            bleu = math.exp(sum(math.log(s) for s in scores) / len(scores))
        else:
            bleu = 0.0
            
        total_scores.append(bleu)
    
    return sum(total_scores) / len(total_scores) * 100


class EvaluationMetrics:
    """
    Container for evaluation metrics.
    """
    
    def __init__(self):
        """Initialize metrics container."""
        self.metrics = {
            'loss': [],
            'perplexity': [],
            'accuracy': [],
            'top_k_accuracy': []
        }
        
    def add_batch_metrics(
        self,
        loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
    ):
        """
        Add metrics from a batch.
        
        Args:
            loss: Batch loss
            predictions: Predicted logits
            targets: Target token IDs
            k: Top-k value for accuracy
        """
        self.metrics['loss'].append(loss)
        self.metrics['perplexity'].append(calculate_perplexity(loss))
        self.metrics['accuracy'].append(
            calculate_accuracy(predictions, targets)
        )
        self.metrics['top_k_accuracy'].append(
            calculate_top_k_accuracy(predictions, targets, k)
        )
        
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average of all metrics.
        
        Returns:
            Dictionary of averaged metrics
        """
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in self.metrics.items()
        }
        
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []
            
    def __str__(self) -> str:
        """String representation of metrics."""
        avg_metrics = self.get_average_metrics()
        return (
            f"Loss: {avg_metrics['loss']:.4f} | "
            f"Perplexity: {avg_metrics['perplexity']:.2f} | "
            f"Accuracy: {avg_metrics['accuracy']:.2f}% | "
            f"Top-5 Acc: {avg_metrics['top_k_accuracy']:.2f}%"
        )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = EvaluationMetrics()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Add metrics
            metrics.add_batch_metrics(
                loss=loss.item(),
                predictions=logits,
                targets=labels
            )
    
    return metrics.get_average_metrics()


def compare_models(
    models: Dict[str, torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of model names to models
        dataloader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Dictionary of model names to their metrics
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, dataloader, device)
        results[model_name] = metrics
        
    return results


def print_evaluation_report(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        formatted_name = metric_name.replace('_', ' ').title()
        
        if 'loss' in metric_name.lower():
            print(f"{formatted_name:<30} {value:>10.4f}")
        elif 'perplexity' in metric_name.lower():
            print(f"{formatted_name:<30} {value:>10.2f}")
        elif 'accuracy' in metric_name.lower():
            print(f"{formatted_name:<30} {value:>9.2f}%")
        else:
            print(f"{formatted_name:<30} {value:>10.4f}")
    
    print("=" * 60)