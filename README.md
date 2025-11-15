# Nima 🤖

> **Learn AI by building your own language model from scratch**

A beginner-friendly project for understanding how AI text generation works. Build and train your own small language models using PyTorch.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 What You'll Learn

- **How AI generates text**: Build the core transformer architecture from scratch
- **Training process**: Watch your model learn and improve over time
- **Real experimentation**: Train models on different types of text (Shakespeare, technical docs)
- **Evaluation**: Test how well your model performs and understand the results

## ✨ What's Included

- 🧠 **Complete AI Model**: All the code to build a text-generating AI
- � **Training Pipeline**: Tools to train your model on any text data
- � **Progress Monitoring**: Visual graphs to watch training progress
- 💬 **Chat Interface**: Test your trained model with questions
- 📚 **Learning Examples**: Pre-configured training on Shakespeare and technical content

## 🏗️ Project Structure

```text
├── src/                    # Core AI model code
│   ├── models/            # The actual neural network
│   ├── data/              # Text processing tools
│   ├── training/          # Training logic
│   └── inference/         # Text generation
├── scripts/               # Easy-to-use training scripts
│   ├── train_technical.py # Train on technical content
│   ├── ask_nima.py        # Chat with your model
│   └── prepare_data.py    # Process your text data
├── configs/               # Training settings
├── experiments/           # Your trained models
└── data/                  # Your training text files
```

## 🚀 Quick Start

### Recommended: Docker Compose (Primary Method)

```bash
# Load port assignments from workspace-config
source ../workspace-config/ports/.env.ports

# Start API service with hot reload
docker-compose up api

# Or start all services (API + Frontend + DB)
docker-compose up
```

**Access:**
- Frontend: http://localhost:3002 (if enabled)
- API: http://localhost:8002
- API Docs: http://localhost:8002/docs

**Benefits:**
- Consistent environment (matches production)
- Hot reload enabled via volume mounts
- No local Python version conflicts
- Single command to start everything

See `../workspace-config/docs/DOCKER_COMPOSE_GUIDE.md` for complete guide.

### Alternative: Local Development (Fallback)

#### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

#### Step 2: Start API

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8002
```

#### Step 3: Try Training on Shakespeare

```bash
# Train a small model on Shakespeare text
python scripts/train.py --config configs/base_model.yaml

# Watch training progress (in another terminal)
tensorboard --logdir experiments/logs
```

#### Step 4: Chat with Your Model

```bash
# Test your trained model
python scripts/ask_nima.py \
  --checkpoint experiments/checkpoints/best_model.pt \
  --prompt "To be or not to be"
```

#### Step 5: Train on Your Own Text

```bash
# Train on technical documentation
python scripts/train_technical.py --config configs/technical_training.yaml

# Chat with technical model
python scripts/ask_nima.py \
  --checkpoint experiments/nima_technical/checkpoint_best.pt \
  --prompt "What is Kubernetes?"
```

## 🧠 How It Works

The AI model uses a **transformer architecture** - the same technology behind ChatGPT, but much smaller for learning:

- **Attention Mechanism**: The model learns which words to pay attention to
- **Text Processing**: Converts text into numbers the AI can understand
- **Training Loop**: The model gradually learns patterns in your text data
- **Generation**: Uses learned patterns to create new, similar text

## 🎓 Learning Path

### Beginner (Start Here)

1. **Run the Shakespeare example** - See a working model in 10 minutes
2. **Watch TensorBoard** - Understand how training works visually
3. **Try the chat interface** - See what your model learned
4. **Read the training logs** - Understand loss, perplexity, and improvement

### Intermediate

1. **Train on your own text** - Use technical documentation or your writing
2. **Experiment with settings** - Change model size, learning rate, epochs
3. **Compare different datasets** - See how data affects model quality
4. **Study the code** - Understand attention, embeddings, and training loops

### Advanced

1. **Modify the architecture** - Add layers, change attention heads
2. **Implement new features** - Try different optimizers or sampling methods
3. **Scale up training** - Use larger datasets and models
4. **Fine-tune pre-trained models** - Start with existing models instead of from scratch

## ⚙️ Configuration Files

Training settings are in simple YAML files in `configs/`:

- `base_model.yaml` - Small model, good for learning
- `technical_training.yaml` - Settings for technical content
- `longer_training.yaml` - Train for more epochs

## 📈 Monitoring Your Training

**TensorBoard** shows real-time graphs:

```bash
tensorboard --logdir experiments/logs
# Open http://localhost:6006 in your browser
```

**Training logs** explain what's happening:

- **Loss**: How wrong the model is (lower = better)
- **Perplexity**: How confused the model is (lower = better)
- **Epoch**: One complete pass through your data

## 🎯 Understanding Results

**Good signs**:

- Loss decreases over time
- Generated text makes sense
- Model responds appropriately to prompts

**Warning signs**:

- Loss stops decreasing (model stopped learning)
- Generated text is repetitive or gibberish
- Need more/better training data

## 📚 Learning Resources

**Documentation**: Check the `docs/` folder for detailed guides
**Examples**: Look in `scripts/` for working code examples  
**Configs**: See `configs/` for training settings you can modify

## 💡 Key Machine Learning Concepts You'll Learn

- **Neural Networks**: How layers of math create intelligence
- **Transformers**: The architecture that powers modern AI
- **Training**: How models learn from data through repetition
- **Overfitting**: When models memorize instead of learning
- **Evaluation**: How to measure if your model is working
- **Hyperparameters**: Settings that control learning speed and quality

## 🚀 Next Steps for Learning

1. **Start small**: Train the Shakespeare model first
2. **Understand the basics**: Learn what loss and perplexity mean
3. **Experiment**: Try different settings and see what happens
4. **Read the code**: Start with `src/models/` to understand the AI architecture
5. **Scale up**: Try larger models and datasets as you learn more

This project gives you hands-on experience with the same concepts used in ChatGPT, just at a smaller, learnable scale.

Happy learning! 🎓
