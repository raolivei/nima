# Changelog

All notable changes to the NIMA LLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-11-14

### Added
- React frontend application with Vite build system
- Dark/light mode toggle with persistent theme storage
- Elegant, modern, arty robot aesthetic design
- Geometric robot logo with animated pulsing eyes
- Comprehensive responsive design for mobile, tablet, and desktop
- API URL configuration input
- Real-time status messages and error handling
- Smooth animations and transitions throughout

### Changed
- Complete UI redesign with minimal monochrome aesthetic
- Updated ingress configuration to use `nima.eldertree.local` domain
- Improved alignment and borders for all screen sizes
- Enhanced typography with generous letter spacing
- Sharp, angular design language throughout

### Fixed
- Responsive layout issues on mobile devices
- Border and alignment inconsistencies across screen sizes
- Container padding and spacing optimizations

## [0.3.1] - 2025-01-XX

### Added
- Dockerfile for containerized deployment
- FastAPI application (api/main.py) for serving the model
- Kubernetes deployment manifests:
  - deploy.yaml - Deployment configuration
  - service.yaml - Service configuration
  - ingress.yaml - Ingress configuration
  - namespace.yaml - Namespace definition
  - pvc.yaml - Persistent volume claim
  - secret.yaml.example - Secret template

## [0.3.0] - 2025-10-15

### Added

- Comprehensive project validation and testing framework
- Interactive ask_nima.py script for model interaction
- Enhanced PyTorch 2.6+ compatibility with weights_only parameter handling
- TokenizerAdapter for seamless inference pipeline integration

### Fixed

- PyTorch checkpoint loading compatibility issues
- TrainingConfig attribute errors in training pipeline
- Circular import issues in training modules

### Documentation

- Created technical article for Medium publication
- Updated documentation with practical examples
- Enhanced architecture explanations with code samples

## [0.2.0] - 2025-10-14

### Added

- **Complete Training System**: Added full pipeline for training AI models from scratch
- **Interactive Chat**: Test your trained models with `scripts/ask_nima.py`
- **Smart Training**: Automatic early stopping when model isn't improving
- **Visual Monitoring**: TensorBoard integration to watch training progress
- **Technical Focus**: Special tools for training on technical documentation

### Major Files Added

- `scripts/train_technical.py` - Train models on technical content
- `scripts/ask_nima.py` - Chat with your trained models
- `configs/technical_training.yaml` - Easy configuration file
- `src/training/monitoring.py` - Training progress tracking

### What Works Now

- Train small models on your own data (like Shakespeare or technical docs)
- Watch training progress in real-time with graphs
- Chat with trained models to test quality
- Automatic saving of best model versions

### Learning Notes

**Current Results**: Successfully trained a small model (3.3M parameters) that can generate text. Training loss improved from 4.85 to 3.76 over 10 epochs.

**Key Lesson**: Model quality depends heavily on dataset size. Our technical model (18 samples) produces lower quality text than the Shakespeare model (much larger dataset).

**Next Steps**: Either collect more training data or learn about fine-tuning pre-trained models for better results with small datasets.

## [0.1.0] - 2025-10-11

### Added

- Initial LLM implementation with GPT-style transformer architecture
- Complete data processing pipeline with multiple tokenizer support
- Training system with monitoring and checkpointing
- Basic inference engine for text generation
- Project structure and configuration management

### Features

- 547K parameter transformer model with multi-head attention
- Character-level and BPE tokenization
- Training on Shakespeare and technical datasets
- Model evaluation and text generation capabilities
