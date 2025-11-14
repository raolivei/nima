#!/usr/bin/env python3
"""
Quick Start Example: Train Nima on Technical Content

This script demonstrates the complete workflow for training Nima:
1. Prepare technical data
2. Configure training
3. Train the model
4. Evaluate and generate samples
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 80)
print("NIMA - TECHNICAL MODEL TRAINING")
print("Quick Start Example")
print("=" * 80)

# Check if sample data exists
data_dir = Path("data/raw")
sample_doc = data_dir / "sample_k8s_doc.md"
sample_qa = data_dir / "technical_qa.json"

if not sample_doc.exists() or not sample_qa.exists():
    print("\n⚠️  Sample data not found!")
    print("Please ensure these files exist:")
    print(f"  - {sample_doc}")
    print(f"  - {sample_qa}")
    sys.exit(1)

print("\n✓ Sample data found")
print(f"  - {sample_doc}")
print(f"  - {sample_qa}")

# Step 1: Prepare Data
print("\n" + "=" * 80)
print("STEP 1: Preparing Technical Data")
print("=" * 80)

print("\nRun this command to prepare your data:")
print("\npython scripts/prepare_technical_data.py \\")
print("  --output-dir data/processed/technical_example \\")
print("  --tokenizer bpe \\")
print(f"  --text-files {sample_doc} \\")
print(f"  --json-files {sample_qa} \\")
print("  --format qa \\")
print("  --max-length 512")

print("\nThis will create:")
print("  ✓ Training set (80% of data)")
print("  ✓ Validation set (10% of data)")
print("  ✓ Test set (10% of data)")
print("  ✓ BPE tokenizer")

# Step 2: Configure Training
print("\n" + "=" * 80)
print("STEP 2: Configure Training")
print("=" * 80)

print("\nEdit configs/technical_training.yaml:")
print("\ndata:")
print("  train_file: 'data/processed/technical_example/train.txt'")
print("  val_file: 'data/processed/technical_example/val.txt'")
print("  test_file: 'data/processed/technical_example/test.txt'")
print("  tokenizer_path: 'data/processed/technical_example/tokenizer_bpe.json'")
print("\nmodel:")
print("  preset: 'gpt-tiny'  # Start small for testing")
print("\ntraining:")
print("  epochs: 10")
print("  batch_size: 8")
print("  early_stopping:")
print("    enabled: true")
print("    patience: 3")

# Step 3: Train
print("\n" + "=" * 80)
print("STEP 3: Train the Model")
print("=" * 80)

print("\nRun training:")
print("\npython scripts/train_technical.py \\")
print("  --config configs/technical_training.yaml")

print("\nMonitor training:")
print("  - Watch console output for loss and perplexity")
print("  - View plots in: experiments/nima_technical/plots/")
print("  - TensorBoard: tensorboard --logdir experiments/nima_technical/tensorboard")

# Step 4: Evaluate
print("\n" + "=" * 80)
print("STEP 4: Evaluate and Generate")
print("=" * 80)

print("\nEvaluate on test set:")
print("\npython scripts/train_technical.py \\")
print("  --config configs/technical_training.yaml \\")
print("  --resume experiments/nima_technical/checkpoint_best.pt \\")
print("  --eval-only")

print("\nInteractive generation:")
print("\npython scripts/inference.py \\")
print("  --checkpoint experiments/nima_technical/checkpoint_best.pt \\")
print("  --tokenizer data/processed/technical_example/tokenizer_bpe.json \\")
print("  --mode interactive")

# Expected Results
print("\n" + "=" * 80)
print("EXPECTED RESULTS")
print("=" * 80)

print("\n📊 Training Metrics:")
print("  - Initial loss: ~4.0-5.0")
print("  - Final loss: ~2.0-3.0 (depends on data size)")
print("  - Perplexity should decrease over time")
print("  - Validation loss should track training loss")

print("\n💡 Sample Generations:")
print("  Prompt: 'Kubernetes is'")
print("  Expected: Technical explanation about Kubernetes")
print("\n  Prompt: 'To deploy with Terraform'")
print("  Expected: Instructions for Terraform deployment")

print("\n⚡ Performance:")
print("  - gpt-tiny: ~5-10 minutes on CPU")
print("  - gpt-small: ~30-60 minutes on CPU, ~5-10 minutes on GPU")
print("  - Early stopping will reduce training time")

# Troubleshooting
print("\n" + "=" * 80)
print("TROUBLESHOOTING")
print("=" * 80)

print("\n❌ If you see 'Out of Memory':")
print("  - Reduce batch_size to 4 or 2")
print("  - Use gpt-tiny instead of gpt-small")
print("  - Reduce max_length to 256")

print("\n❌ If loss doesn't decrease:")
print("  - Check data was prepared correctly")
print("  - Verify tokenizer vocabulary size > 100")
print("  - Increase warmup_steps to 500")
print("  - Lower learning_rate to 1e-4")

print("\n❌ If training is too slow:")
print("  - Use GPU if available")
print("  - Increase batch_size")
print("  - Use gpt-tiny for testing")
print("  - Reduce max_length")

# Next Steps
print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\n1. Add more data:")
print("   - Collect more technical documentation")
print("   - Add Q&A datasets")
print("   - Include code examples")

print("\n2. Experiment with model size:")
print("   - Try gpt-small for better quality")
print("   - Adjust layers and dimensions")

print("\n3. Tune hyperparameters:")
print("   - Learning rate: 1e-4 to 5e-4")
print("   - Batch size: 8 to 32")
print("   - Warmup steps: 500 to 2000")

print("\n4. Evaluate quality:")
print("   - Generate samples with different prompts")
print("   - Check technical accuracy")
print("   - Test on domain-specific questions")

print("\n" + "=" * 80)
print("For detailed documentation, see:")
print("  - docs/training_technical.md")
print("  - docs/architecture.md")
print("  - docs/getting_started.md")
print("=" * 80 + "\n")

print("🚀 Ready to start! Run the commands above to train Nima.\n")
