# Architecture Guide

This guide explains the transformer architecture implemented in Nima and how the different components work together.

## Overview

Nima implements a complete transformer-based language model from scratch, following the "Attention Is All You Need" paper with modern improvements.

## Core Components

### 1. Attention Mechanism

The attention mechanism is the heart of the transformer. It allows the model to focus on different parts of the input when processing each token.

#### Scaled Dot-Product Attention

```python
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

- **Q (Query)**: What we're looking for
- **K (Key)**: What we're looking at
- **V (Value)**: What we actually get
- **d_k**: Dimension of keys (for scaling)

#### Multi-Head Attention

Instead of one attention mechanism, we use multiple parallel attention "heads":

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Benefits:
- Captures different types of relationships
- Attends to different positions simultaneously
- More expressive representation

### 2. Feed-Forward Networks

After attention, we apply a position-wise feed-forward network:

```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Nima implements three variants:

1. **Standard FFN**: ReLU activation
2. **GLU**: Gated Linear Unit
3. **SwiGLU**: Swish-Gated Linear Unit (modern, used in LLaMA)

### 3. Positional Encoding

Transformers have no inherent notion of position, so we add positional information:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This creates a unique encoding for each position that the model can learn to use.

### 4. Layer Normalization

Applied before each sub-layer (pre-norm):

```python
output = LayerNorm(x + Sublayer(x))
```

Benefits:
- Stabilizes training
- Allows deeper models
- Faster convergence

## Model Architectures

### GPT-Style (Decoder-Only)

The GPT architecture is designed for autoregressive text generation:

```
Input → Token Embedding + Position Embedding
  ↓
Decoder Layer 1: Masked Self-Attention → FFN
  ↓
Decoder Layer 2: Masked Self-Attention → FFN
  ↓
  ...
  ↓
Decoder Layer N: Masked Self-Attention → FFN
  ↓
Layer Norm → Output Projection → Logits
```

Key features:
- **Causal masking**: Can only attend to previous tokens
- **Autoregressive**: Generates one token at a time
- **Unidirectional**: Left-to-right processing

### Transformer (Encoder-Decoder)

The full transformer includes both encoder and decoder:

```
Source → Encoder → Context
Target → Decoder (with cross-attention to Context) → Output
```

Useful for:
- Machine translation
- Summarization
- Any seq2seq task

## Training Process

### Forward Pass

1. **Embedding**: Convert token IDs to vectors
2. **Position encoding**: Add positional information
3. **Transformer layers**: Process through N layers
4. **Output projection**: Map to vocabulary logits
5. **Loss calculation**: Cross-entropy with targets

### Backward Pass

1. **Gradient computation**: Backprop through all layers
2. **Gradient clipping**: Prevent exploding gradients
3. **Optimizer step**: Update weights (Adam)
4. **Learning rate scheduling**: Adjust LR over time

### Learning Rate Schedule

Nima uses cosine annealing with warmup:

```
Linear warmup: LR increases from 0 to max_lr
Cosine decay: LR decreases following cosine curve
```

Benefits:
- Warm start prevents instability
- Cosine decay allows fine-tuning at the end

## Memory and Efficiency

### Attention Complexity

Self-attention has O(n²) complexity where n is sequence length:

```
Memory: O(batch_size × n² × d_model)
Compute: O(batch_size × n² × d_model)
```

This is why we limit max_seq_len (typically 512-2048).

### Optimization Techniques

1. **Gradient Accumulation**: Simulate larger batches
2. **Mixed Precision**: Use FP16 for speed
3. **Gradient Checkpointing**: Trade compute for memory
4. **Model Parallelism**: Split large models across devices

## Text Generation

### Sampling Strategies

#### 1. Greedy Decoding
```python
next_token = argmax(logits)
```
- Deterministic
- Fast
- Can be repetitive

#### 2. Temperature Sampling
```python
probs = softmax(logits / temperature)
next_token = sample(probs)
```
- Temperature < 1: More focused
- Temperature > 1: More random

#### 3. Top-K Sampling
```python
top_k_probs = keep_top_k(probs, k)
next_token = sample(top_k_probs)
```
- Only sample from k most likely tokens
- Prevents unlikely tokens

#### 4. Top-P (Nucleus) Sampling
```python
top_p_probs = keep_cumulative(probs, p)
next_token = sample(top_p_probs)
```
- Dynamic cutoff based on cumulative probability
- Adapts to confidence

#### 5. Beam Search
```python
Maintain k best sequences
Expand each by vocabulary
Keep top k total
```
- More thorough search
- Better for tasks with clear correct answer
- Slower than sampling

## Model Scaling

### Compute-Optimal Scaling

Based on the Chinchilla paper, for a given compute budget:

```
Model size ∝ sqrt(compute)
Training tokens ∝ sqrt(compute)
```

In practice:
- **Small models**: Train longer (more data)
- **Large models**: Need proportionally more data

### Nima Model Sizes

| Preset | Parameters | d_model | Layers | Heads | FFN | Use Case |
|--------|------------|---------|--------|-------|-----|----------|
| tiny   | ~10M       | 256     | 4      | 4     | 1024 | Testing |
| small  | ~125M      | 768     | 12     | 12    | 3072 | Experiments |
| medium | ~350M      | 1024    | 24     | 16    | 4096 | Small projects |

## Best Practices

### Architecture Choices

1. **Model size**: Start small, scale up if needed
2. **Sequence length**: Longer = more context but more memory
3. **Vocabulary size**: Balance coverage vs. model size
4. **Number of layers**: More layers = more capacity

### Training Tips

1. **Learning rate**: Most important hyperparameter
2. **Batch size**: Larger = more stable, needs more memory
3. **Warmup steps**: Typically 1-10% of total steps
4. **Evaluation frequency**: Balance speed vs. monitoring

### Generation Tips

1. **Temperature**: 0.7-0.9 for creative text
2. **Top-p**: 0.9-0.95 is usually good
3. **Repetition penalty**: 1.2 helps prevent loops
4. **Max length**: Set appropriately for your task

## Advanced Topics

### Positional Encodings

Alternatives to sinusoidal:

1. **Learned positions**: Train embedding for each position
2. **Relative positions**: Distance-based attention
3. **RoPE**: Rotary position embeddings (used in LLaMA)
4. **ALiBi**: Attention with linear biases

### Attention Variants

1. **Flash Attention**: Memory-efficient attention
2. **Sparse Attention**: Reduced complexity for long sequences
3. **Cross Attention**: Attend to encoder output (in decoder)
4. **Local Attention**: Only attend to nearby tokens

### Model Improvements

1. **Layer normalization placement**: Pre-norm vs post-norm
2. **Activation functions**: ReLU, GELU, SwiGLU
3. **Residual connections**: Skip connections prevent gradient vanishing
4. **Weight tying**: Share embedding and output weights

## Implementation Details

### Masking

```python
# Causal mask (for GPT)
mask = torch.tril(torch.ones(seq_len, seq_len))

# Padding mask
mask = (tokens != pad_token_id)

# Combined
final_mask = causal_mask & padding_mask
```

### Initialization

```python
# Xavier initialization for linear layers
nn.init.xavier_uniform_(linear.weight)

# Small random for embeddings
nn.init.normal_(embedding.weight, mean=0, std=0.02)
```

### Gradient Clipping

```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

Prevents exploding gradients in deep networks.

## Further Reading

- **Original paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT**: "Improving Language Understanding by Generative Pre-Training"
- **Scaling laws**: "Training Compute-Optimal Large Language Models" (Chinchilla)
- **Modern techniques**: LLaMA, PaLM architecture papers

## Summary

Nima implements a complete, modern transformer architecture with:

- ✅ Multi-head attention with proper scaling
- ✅ Multiple feed-forward variants
- ✅ Proper normalization and residual connections  
- ✅ Flexible positional encoding
- ✅ Efficient training infrastructure
- ✅ Advanced generation techniques

The architecture is modular and extensible, making it easy to experiment with variations and improvements!
