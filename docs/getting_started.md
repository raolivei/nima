# Getting Started with Nima

Welcome to Nima! This guide will help you get started with building and training your own Large Language Model from scratch.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/raolivei/nima.git
cd nima

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your First Dataset

```bash
# Download and prepare Tiny Shakespeare dataset
python scripts/prepare_data.py --dataset tiny_shakespeare --tokenizer char
```

This will:
- Download the Tiny Shakespeare dataset
- Create a character-level tokenizer
- Split data into train/validation sets
- Save processed data to `data/processed/`

### 3. Train Your First Model

```bash
# Quick test training (2 epochs, small model)
python scripts/train.py --quick_test

# Full training with config
python scripts/train.py --config configs/base_model.yaml
```

### 4. Generate Text

```bash
# Generate text from trained model
python scripts/inference.py \
    --checkpoint experiments/checkpoints/model_latest.pt \
    --tokenizer data/processed/tiny_shakespeare/tokenizer_char.json \
    --prompt "To be or not to be" \
    --mode generate
```

## Understanding the Components

### Models

Nima includes several transformer architectures:

- **GPT-style models**: Decoder-only architecture for text generation
- **Transformer models**: Full encoder-decoder for seq2seq tasks

Available model sizes:
- `gpt-tiny`: 256d, 4 layers (good for learning/testing)
- `gpt-small`: 768d, 12 layers (GPT-1 style)
- `gpt-medium`: 1024d, 24 layers (GPT-2 medium style)

### Tokenizers

Three tokenization strategies are available:

1. **Character-level** (`char`):
   - Simplest approach
   - Good for small datasets
   - Large vocabulary for general text

2. **Word-level** (`word`):
   - Splits on words
   - Human-readable tokens
   - Better for well-formed text

3. **BPE** (`bpe`):
   - Subword tokenization
   - Balanced vocabulary size
   - Handles rare words well

### Data Processing

The data pipeline handles:

- **Downloading**: Common datasets (Shakespeare, WikiText, etc.)
- **Preprocessing**: Text cleaning, normalization
- **Tokenization**: Converting text to token IDs
- **Batching**: Creating efficient training batches

## Your First Training Run

Let's walk through a complete training example:

### Step 1: Prepare Data

```bash
python scripts/prepare_data.py \
    --dataset tiny_shakespeare \
    --tokenizer char \
    --vocab_size 10000 \
    --data_dir data
```

Output:
```
Dataset prepared:
  Tokenizer: char
  Vocabulary size: 65
  Training sequences: 8,432
  Validation sequences: 937
```

### Step 2: Configure Training

Create or edit `configs/my_model.yaml`:

```yaml
model:
  vocab_size: 65  # From tokenizer
  d_model: 256
  n_heads: 4
  n_layers: 4
  d_ff: 1024
  max_seq_len: 512
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.0003
  num_epochs: 50
  warmup_steps: 1000
  save_every: 1000
  eval_every: 500
```

### Step 3: Start Training

```bash
python scripts/train.py --config configs/my_model.yaml
```

You'll see output like:
```
Epoch 1/50
  Step 100: loss=3.2456, lr=0.000030
  Step 200: loss=2.8934, lr=0.000060
  ...
  Validation: loss=2.7123, perplexity=15.07
```

### Step 4: Monitor Progress

Training logs are saved to `experiments/logs/`. You can monitor with TensorBoard:

```bash
tensorboard --logdir experiments/logs
```

### Step 5: Generate Text

```bash
python scripts/inference.py \
    --checkpoint experiments/checkpoints/model_epoch_10.pt \
    --tokenizer data/processed/tiny_shakespeare/tokenizer_char.json \
    --prompt "ROMEO:" \
    --max_length 200 \
    --temperature 0.8
```

## Next Steps

Now that you've run your first model, here are some things to try:

### Experiment with Hyperparameters

- **Model size**: Try different d_model, n_layers
- **Learning rate**: Experiment with different rates and schedules
- **Batch size**: Larger batches = more stable, smaller = more updates

### Try Different Datasets

```bash
# Use WikiText-2
python scripts/prepare_data.py --dataset wikitext2 --tokenizer bpe

# Use your own text file
python scripts/prepare_data.py --dataset /path/to/your/book.txt --tokenizer word
```

### Advanced Generation

```bash
# Interactive mode
python scripts/inference.py \
    --checkpoint model.pt \
    --tokenizer tokenizer.json \
    --mode interactive

# Compare sampling strategies
python scripts/inference.py \
    --checkpoint model.pt \
    --tokenizer tokenizer.json \
    --mode compare \
    --prompt "Once upon a time"
```

## Common Issues

### Out of Memory

If you get CUDA out of memory errors:

1. Reduce batch size in config
2. Reduce model size (d_model, n_layers)
3. Reduce max_seq_len
4. Enable gradient accumulation

### Poor Generation Quality

If generated text is incoherent:

1. Train for more epochs
2. Use a larger model
3. Use more/better training data
4. Adjust temperature (lower = more conservative)

### Slow Training

To speed up training:

1. Use GPU if available
2. Increase batch size (if memory allows)
3. Use fewer evaluation steps
4. Reduce model size for quick experiments

## Understanding the Output

### Training Metrics

- **Loss**: Lower is better (how wrong the predictions are)
- **Perplexity**: exp(loss), represents uncertainty
- **Learning rate**: Current learning rate (varies with schedule)

### Generation Parameters

- **Temperature**: Controls randomness (0.7-1.0 typical)
- **Top-k**: Keep only top k probable tokens (50 typical)
- **Top-p**: Keep tokens with cumulative probability p (0.95 typical)

## Project Structure

```
nima/
├── configs/          # Model and training configurations
├── data/            # Raw and processed datasets
├── experiments/     # Training outputs (checkpoints, logs)
├── scripts/         # Executable scripts
├── src/             # Core library code
│   ├── models/      # Model architectures
│   ├── data/        # Data processing
│   ├── training/    # Training framework
│   ├── evaluation/  # Evaluation metrics
│   └── inference/   # Text generation
└── docs/            # Documentation
```

## Getting Help

- Check the documentation in `docs/`
- Review example scripts in `scripts/`
- Experiment with the demo notebooks in `notebooks/`

Happy building! 🚀