# Training Nima for Technical Knowledge

This guide walks you through training Nima as a specialized model focused on system engineering, DevOps, and technical documentation.

## Overview

Nima can be trained on technical content including:

- Kubernetes, Terraform, AWS documentation
- Python, Go, and other programming language guides
- Engineering blogs and tutorials
- Q&A datasets (StackOverflow-style)
- Custom curated notes and transcripts

## Quick Start

### 1. Prepare Your Data

```bash
# Prepare technical documentation
python scripts/prepare_technical_data.py \
  --output-dir data/processed/technical \
  --tokenizer bpe \
  --text-files docs/kubernetes.md docs/terraform.md \
  --json-files data/raw/qa_dataset.json \
  --format qa \
  --max-length 512
```

This will create:

- `data/processed/technical/train.txt` (80% of data)
- `data/processed/technical/val.txt` (10% of data)
- `data/processed/technical/test.txt` (10% of data)
- `data/processed/technical/tokenizer_bpe.json`
- `data/processed/technical/dataset_metadata.json`

### 2. Configure Training

Edit `configs/technical_training.yaml` to customize:

```yaml
# Model size (choose one)
model:
  preset: "gpt-small" # or "gpt-tiny", "gpt-medium"

# Training parameters
training:
  epochs: 50
  batch_size: 16
  learning_rate: 3.0e-4
  warmup_steps: 2000

  # Early stopping
  early_stopping:
    enabled: true
    patience: 5
    metric: "val_loss"
```

### 3. Start Training

```bash
# Train with monitoring
python scripts/train_technical.py \
  --config configs/technical_training.yaml
```

### 4. Monitor Training

**TensorBoard:**

```bash
tensorboard --logdir experiments/nima_technical/tensorboard
```

**Weights & Biases (optional):**
Enable in config:

```yaml
monitoring:
  wandb:
    enabled: true
    project: "nima-technical"
```

### 5. Evaluate on Test Set

```bash
# Evaluate best checkpoint
python scripts/train_technical.py \
  --config configs/technical_training.yaml \
  --resume experiments/nima_technical/checkpoint_best.pt \
  --eval-only
```

## Data Preparation Details

### Supported Data Formats

#### 1. Plain Text / Markdown

```bash
--text-files documentation.txt guide.md README.md
```

#### 2. Q&A JSON

```json
[
  {
    "question": "How do I deploy with Kubernetes?",
    "answer": "To deploy with Kubernetes, you create a Deployment..."
  }
]
```

```bash
--json-files qa_data.json --format qa
```

#### 3. Chat/Conversation JSON

```json
[
  {
    "messages": [
      { "role": "user", "content": "Explain Docker" },
      { "role": "assistant", "content": "Docker is..." }
    ]
  }
]
```

```bash
--json-files conversations.json --format chat
```

#### 4. Document JSON

```json
[
  {
    "title": "Kubernetes Architecture",
    "content": "Kubernetes follows a master-worker architecture..."
  }
]
```

```bash
--json-files documents.json --format doc
```

#### 5. JSONL (one JSON per line)

```bash
--jsonl-files data.jsonl --format qa
```

### Data Preprocessing Features

The preprocessor automatically:

- ✅ Preserves code blocks (`code`)
- ✅ Maintains technical formatting
- ✅ Keeps command-line examples
- ✅ Removes excessive whitespace
- ✅ Filters samples by length (min/max)
- ✅ Shuffles and splits (80/10/10)

## Training Configuration

### Model Presets

Choose based on your resources:

| Preset     | Parameters | Size   | GPU RAM | Training Time |
| ---------- | ---------- | ------ | ------- | ------------- |
| gpt-tiny   | 4.6M       | 17 MB  | 2 GB    | Fast          |
| gpt-small  | 90M        | 342 MB | 8 GB    | Medium        |
| gpt-medium | 350M       | 1.3 GB | 16 GB   | Slow          |

### Training Parameters

**Learning Rate Schedule:**

```yaml
training:
  learning_rate: 3.0e-4
  warmup_steps: 2000 # Linear warmup
  lr_scheduler: "cosine" # Cosine decay
  min_lr: 1.0e-5
```

**Batch Size & Gradient Accumulation:**

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 4
  # Effective batch size = 16 * 4 = 64
```

**Early Stopping:**

```yaml
training:
  early_stopping:
    enabled: true
    patience: 5 # Stop after 5 evals without improvement
    min_delta: 0.001
    metric: "val_loss"
    mode: "min"
```

### Monitoring & Logging

**Track Metrics:**

- Training loss and perplexity
- Validation loss and perplexity
- Learning rate
- Gradient norms

**Automatic Plots:**

- Loss curves (train vs val)
- Perplexity over time
- Learning rate schedule

Saved to: `experiments/nima_technical/plots/`

## Training Tips

### 1. Start Small

```bash
# Use tiny model first to verify pipeline
model:
  preset: "gpt-tiny"

training:
  epochs: 5
```

### 2. Monitor Overfitting

Watch for validation loss increasing while training loss decreases:

```yaml
early_stopping:
  enabled: true
  patience: 3
```

### 3. Adjust Learning Rate

If loss plateaus:

```yaml
training:
  learning_rate: 1.0e-4 # Lower LR
  warmup_steps: 4000 # Longer warmup
```

### 4. Handle Limited GPU Memory

```yaml
training:
  batch_size: 4 # Smaller batches
  gradient_accumulation_steps: 16 # More accumulation
  use_mixed_precision: true # FP16 training
```

### 5. Use Checkpointing

Best model automatically saved to:

```
experiments/nima_technical/checkpoint_best.pt
```

Resume training:

```bash
python scripts/train_technical.py \
  --config configs/technical_training.yaml \
  --resume experiments/nima_technical/checkpoint_best.pt
```

## Evaluation

### Automatic Metrics

- **Loss**: Cross-entropy loss
- **Perplexity**: exp(loss), lower is better
- **Accuracy**: Token-level accuracy

### Sample Generation

Configure in `configs/technical_training.yaml`:

```yaml
evaluation:
  generation:
    enabled: true
    prompts:
      - "Kubernetes is"
      - "To deploy with Terraform"
      - "Docker containers"
    max_length: 100
    temperature: 0.8
    top_k: 50
    top_p: 0.95
    num_samples: 3
```

## Example Workflow

### Complete Training Pipeline

```bash
# 1. Prepare data from multiple sources
python scripts/prepare_technical_data.py \
  --output-dir data/processed/k8s_training \
  --tokenizer bpe \
  --text-files docs/k8s/*.md \
  --json-files data/raw/k8s_qa.json \
  --format qa \
  --max-length 512

# 2. Train model with monitoring
python scripts/train_technical.py \
  --config configs/technical_training.yaml

# 3. Evaluate on test set
python scripts/train_technical.py \
  --config configs/technical_training.yaml \
  --resume experiments/nima_technical/checkpoint_best.pt \
  --eval-only

# 4. Interactive testing
python scripts/inference.py \
  --checkpoint experiments/nima_technical/checkpoint_best.pt \
  --tokenizer data/processed/k8s_training/tokenizer_bpe.json \
  --mode interactive
```

## Advanced Features

### Multi-Dataset Training

Weight different data sources:

```python
sources = [
    DataSource("kubernetes.md", "markdown", weight=2.0),
    DataSource("terraform.md", "markdown", weight=1.5),
    DataSource("qa_data.json", "json", format="qa", weight=1.0)
]
```

### Custom Tokenizer Vocabulary

```bash
# Build larger BPE vocabulary
python scripts/prepare_technical_data.py \
  --tokenizer bpe \
  --max-length 1024  # Longer sequences
```

### Distributed Training

```yaml
hardware:
  distributed: true
  gpu_ids: [0, 1, 2, 3]
  backend: "nccl"
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Enable `use_mixed_precision: true`
- Use smaller model preset

### Loss Not Decreasing

- Increase `warmup_steps`
- Lower `learning_rate`
- Check data quality
- Verify tokenizer vocabulary

### Validation Loss Increasing (Overfitting)

- Enable early stopping
- Reduce model size
- Add more training data
- Increase dropout

### Training Too Slow

- Increase `batch_size`
- Reduce `max_length`
- Use GPU if available
- Enable mixed precision

## Next Steps

1. **Fine-tune on specific domain**: Prepare domain-specific data
2. **Experiment with prompts**: Test different prompt formats
3. **Adjust hyperparameters**: Tune learning rate, batch size
4. **Evaluate quality**: Generate samples, check coherence
5. **Iterate**: Gather more data, retrain

## Resources

- Model architecture: `docs/architecture.md`
- General getting started: `docs/getting_started.md`
- Inference guide: `scripts/inference.py --help`
- Example configs: `configs/`

## Support

For issues or questions:

1. Check logs in `experiments/nima_technical/logs/`
2. Review TensorBoard metrics
3. Verify data preparation output
4. Test with smaller dataset first
