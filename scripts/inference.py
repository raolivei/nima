#!/usr/bin/env python3
"""
Inference script for Nima LLM.

This script demonstrates how to:
1. Load a trained model
2. Generate text with different sampling strategies
3. Run interactive text generation
4. Evaluate model performance
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    from models import GPTModel, create_model_from_preset
    from data import create_tokenizer, prepare_dataset, create_dataloader, DataCollator
    from inference import TextGenerator, InteractiveGenerator
    from evaluation import evaluate_model, print_evaluation_report
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Inference modules not available: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    INFERENCE_AVAILABLE = False


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    """
    Load trained model and tokenizer.
    
    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Load tokenizer
    tokenizer_type = 'char'  # Default, should be specified in config
    if 'word' in tokenizer_path:
        tokenizer_type = 'word'
    elif 'bpe' in tokenizer_path:
        tokenizer_type = 'bpe'
    
    tokenizer = create_tokenizer(tokenizer_type)
    tokenizer.load(tokenizer_path)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model_config = checkpoint.get('config', {})
    model = create_model_from_preset(
        'gpt-tiny',
        vocab_size=tokenizer.vocab_size,
        **model_config
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown')}")
    
    return model, tokenizer, device


def generate_text(
    model,
    tokenizer,
    device,
    prompts: list,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    num_samples: int = 3
):
    """
    Generate text from prompts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
        prompts: List of prompt strings
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p (nucleus) sampling
        num_samples: Number of samples per prompt
    """
    print("\n" + "=" * 80)
    print("Text Generation")
    print("=" * 80)
    
    generator = TextGenerator(model, tokenizer, device)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        
        generated_texts = generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_samples,
            do_sample=True
        )
        
        for i, text in enumerate(generated_texts, 1):
            print(f"\nSample {i}:")
            print(text)
            print()


def interactive_mode(model, tokenizer, device):
    """
    Run interactive text generation mode.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
    """
    print("\n" + "=" * 80)
    print("Interactive Generation Mode")
    print("=" * 80)
    print("Type 'quit' to exit, 'reset' to clear context")
    print()
    
    generator = TextGenerator(model, tokenizer, device)
    interactive = InteractiveGenerator(generator)
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == 'quit':
                print("Goodbye!")
                break
            
            if prompt.lower() == 'reset':
                interactive.reset_context()
                print("Context reset.")
                continue
            
            response = interactive.generate_response(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            
            print(f"Nima: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def evaluate_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    data_path: str
):
    """
    Evaluate a trained checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        data_path: Path to evaluation data
    """
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, tokenizer_path)
    
    # Load evaluation data
    print(f"\nLoading evaluation data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        eval_text = f.read()
    
    from data import TextDataset
    eval_dataset = TextDataset(
        [eval_text],
        tokenizer,
        max_length=512
    )
    
    collator = DataCollator(tokenizer)
    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collator
    )
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(model, eval_loader, device)
    
    # Print results
    print_evaluation_report(metrics, "Evaluation Results")


def demo_sampling_strategies(model, tokenizer, device, prompt: str):
    """
    Demonstrate different sampling strategies.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
        prompt: Prompt for generation
    """
    print("\n" + "=" * 80)
    print("Sampling Strategies Comparison")
    print("=" * 80)
    print(f"Prompt: '{prompt}'")
    print()
    
    generator = TextGenerator(model, tokenizer, device)
    
    strategies = [
        ("Greedy", {"do_sample": False}),
        ("Temperature 0.5", {"temperature": 0.5, "do_sample": True}),
        ("Temperature 1.0", {"temperature": 1.0, "do_sample": True}),
        ("Top-k (k=10)", {"top_k": 10, "do_sample": True}),
        ("Top-p (p=0.9)", {"top_p": 0.9, "do_sample": True}),
    ]
    
    for strategy_name, kwargs in strategies:
        print(f"\n{strategy_name}:")
        print("-" * 80)
        
        texts = generator.generate(
            prompt=prompt,
            max_length=50,
            num_return_sequences=1,
            **kwargs
        )
        
        print(texts[0])


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Nima LLM Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer')
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'interactive', 'evaluate', 'compare'],
                       help='Inference mode')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                       help='Generation prompt')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to generate')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='Path to evaluation data')
    
    args = parser.parse_args()
    
    if not INFERENCE_AVAILABLE:
        print("Error: Inference dependencies not available.")
        print("Please run: pip install -r requirements.txt")
        return
    
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(
            args.checkpoint,
            args.tokenizer
        )
        
        # Run selected mode
        if args.mode == 'generate':
            prompts = [args.prompt]
            generate_text(
                model, tokenizer, device, prompts,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_samples=args.num_samples
            )
            
        elif args.mode == 'interactive':
            interactive_mode(model, tokenizer, device)
            
        elif args.mode == 'evaluate':
            if not args.eval_data:
                print("Error: --eval_data required for evaluation mode")
                return
            evaluate_checkpoint(args.checkpoint, args.tokenizer, args.eval_data)
            
        elif args.mode == 'compare':
            demo_sampling_strategies(model, tokenizer, device, args.prompt)
        
        print("\nInference completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()