# Nima 🤖

> **A complete Large Language Model implementation from scratch**

Nima is a production-ready LLM framework built from first principles using PyTorch. It demonstrates transformer architecture, modern training techniques, and efficient inference with a clean, modular design.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🏗️ **Complete Transformer Implementation**: Multi-head attention, feed-forward networks, and positional encoding from scratch
- 🎨 **Multiple Architectures**: GPT-style decoder-only and full encoder-decoder models
- 📊 **Flexible Tokenization**: Character-level, word-level, and BPE tokenizers
- 🎯 **Specialized Training**: Technical documentation, engineering content, and Q&A datasets
- 🚀 **Production Training**: Early stopping, learning rate scheduling, gradient accumulation
- 📈 **Advanced Monitoring**: TensorBoard, W&B integration, automatic visualization
- 💬 **Advanced Generation**: Top-k, top-p, beam search, and temperature sampling
- 🧪 **Comprehensive Evaluation**: Perplexity, accuracy, BLEU score, and sample generation
- 🔧 **Easy to Extend**: Modular design makes experimentation simple

## 🎯 Why Nima?

- **Educational**: Learn LLMs by building one from scratch
- **Practical**: Train real models on your own data
- **Customizable**: Modify any component to experiment with new ideas
- **Well-Documented**: Extensive documentation and examples

## 🏗️ Project Structure

```
├── src/                    # Core implementation
│   ├── models/            # Model architectures
│   ├── data/              # Data processing
│   ├── training/          # Training loops
│   ├── evaluation/        # Evaluation metrics
│   ├── inference/         # Inference engine
│   └── utils/             # Utility functions
├── data/                  # Dataset storage
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── configs/              # Configuration files
├── experiments/          # Training experiments
│   ├── checkpoints/      # Model checkpoints
│   └── logs/             # Training logs
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Basic Training (Tiny Shakespeare)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare sample data
python scripts/prepare_data.py --dataset tiny_shakespeare --tokenizer char

# 3. Quick training test
python scripts/train.py --quick_test

# 4. Generate text
python scripts/inference.py \
  --checkpoint experiments/gpt_model/checkpoint_best.pt \
  --tokenizer data/processed/tiny_shakespeare/char_tokenizer.json \
  --prompt "Once upon a time"
```

### Technical Model Training (System Engineering)

```bash
# 1. View quick start guide
python scripts/example_technical_training.py

# 2. Prepare technical data
python scripts/prepare_technical_data.py \
  --output-dir data/processed/technical \
  --tokenizer bpe \
  --text-files data/raw/sample_k8s_doc.md \
  --json-files data/raw/technical_qa.json \
  --format qa

# 3. Train specialized model
python scripts/train_technical.py \
  --config configs/technical_training.yaml

# 4. Monitor training
tensorboard --logdir experiments/nima_technical/tensorboard

# 5. Evaluate and generate
python scripts/train_technical.py \
  --config configs/technical_training.yaml \
  --resume experiments/nima_technical/checkpoint_best.pt \
  --eval-only
```

## 🧠 Model Architecture

Our implementation includes:

- **Multi-Head Attention**: Core attention mechanism
- **Positional Encoding**: Position-aware embeddings
- **Feed-Forward Networks**: Transformer building blocks
- **Layer Normalization**: Training stability
- **Residual Connections**: Gradient flow optimization

## 📊 Implementation Status

### Core Architecture ✅

- [x] Multi-head attention mechanism
- [x] Multiple transformer architectures (Encoder-Decoder, GPT-style)
- [x] Positional encoding (learned and sinusoidal)
- [x] Layer normalization and residual connections
- [x] Model factory with pre-configured sizes

### Data Processing ✅

- [x] Three tokenizer types (char, word, BPE)
- [x] Efficient data loading and preprocessing
- [x] Technical data preparation pipeline
- [x] Multi-format support (text, markdown, JSON, JSONL)
- [x] 80/10/10 train/val/test splits

### Training ✅

- [x] Training pipeline with checkpointing
- [x] Early stopping
- [x] Learning rate scheduling (warmup + cosine decay)
- [x] Gradient clipping and accumulation
- [x] Mixed precision training (FP16)
- [x] TensorBoard and W&B integration

### Evaluation ✅

- [x] Comprehensive metrics (perplexity, accuracy, BLEU)
- [x] Automatic visualization (loss curves, plots)
- [x] Test set evaluation
- [x] Sample text generation for verification

### Inference ✅

- [x] Advanced text generation (sampling strategies)
- [x] Top-k, top-p (nucleus), temperature sampling
- [x] Beam search
- [x] Interactive generation mode
- [x] Batch generation

### Coming Soon 🚧

- [ ] Distributed training (multi-GPU)
- [ ] Model quantization
- [ ] ONNX export
- [ ] Efficient attention (Flash Attention)
- [ ] Fine-tuning utilities

## 🔧 Configuration

Model and training parameters are managed through YAML configuration files in the `configs/` directory:

- `base_model.yaml`: Basic model configuration
- `small_model.yaml`: Smaller model for quick experimentation
- `large_model.yaml`: Larger model for better performance

## 📈 Monitoring

Training progress can be monitored using:

- **TensorBoard**: Real-time training metrics
- **Weights & Biases**: Experiment tracking (optional)
- **Custom logging**: Detailed training logs

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 🎓 Training Specialized Models

Nima supports training on specialized domains like system engineering and technical documentation:

### Supported Data Types

- **Technical Documentation**: Kubernetes, Terraform, AWS, DevOps guides
- **Q&A Datasets**: StackOverflow-style technical questions and answers
- **Code Examples**: With syntax preservation for various languages
- **Engineering Blogs**: Technical articles and tutorials
- **Custom Notes**: Your own curated technical content

### Key Features

- **Smart Preprocessing**: Preserves code blocks, technical formatting, and commands
- **Multi-Source**: Combine multiple data sources with custom weights
- **Automatic Splits**: 80/10/10 train/validation/test splits with shuffling
- **Early Stopping**: Prevents overfitting with configurable patience
- **Comprehensive Monitoring**: TensorBoard, W&B, and automatic plot generation
- **Sample Generation**: Verify model quality with domain-specific prompts

### Example: Training on Kubernetes Documentation

```bash
# Prepare data
python scripts/prepare_technical_data.py \
  --output-dir data/processed/k8s \
  --tokenizer bpe \
  --text-files docs/k8s/*.md \
  --json-files data/raw/k8s_qa.json \
  --format qa

# Train
python scripts/train_technical.py \
  --config configs/technical_training.yaml

# Generate samples
python scripts/inference.py \
  --checkpoint experiments/nima_technical/checkpoint_best.pt \
  --tokenizer data/processed/k8s/tokenizer_bpe.json \
  --prompt "To deploy with Kubernetes"
```

See **[docs/training_technical.md](docs/training_technical.md)** for complete guide.

## 📚 Learning Resources

### Documentation

- **[Getting Started](docs/getting_started.md)**: Quick start guide with examples
- **[Architecture](docs/architecture.md)**: Deep dive into transformer implementation
- **[Training Guide](docs/training.md)**: General training pipeline
- **[Technical Training](docs/training_technical.md)**: Specialized model training

### Notebooks

Check out the `notebooks/` directory for:

- Architecture deep dives
- Training tutorials
- Inference examples
- Performance analysis

## 🤝 Contributing

This is a learning project! Feel free to:

1. Fork the repository
2. Create feature branches
3. Submit pull requests
4. Share improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Learning Journey

Document your learning process and insights as you build and evolve this LLM. Each component teaches fundamental ML concepts that apply broadly in the field.

Happy coding! 🚀
