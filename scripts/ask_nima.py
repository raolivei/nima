#!/usr/bin/env python3
"""
Simple script to ask questions to Nima.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.transformer import GPTModel
from data.tokenizer import SimpleBPETokenizer, CharTokenizer, WordTokenizer
import json


def load_model(checkpoint_path, tokenizer):
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get vocab size from tokenizer
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
    
    # Try to infer architecture from checkpoint weights if config not available
    if 'token_embedding.weight' in checkpoint['model_state_dict']:
        actual_vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        actual_d_model = checkpoint['model_state_dict']['token_embedding.weight'].shape[1]
        actual_max_seq_len = checkpoint['model_state_dict']['pos_embedding.weight'].shape[0]
        print(f"Detected from checkpoint: vocab_size={actual_vocab_size}, d_model={actual_d_model}, max_seq_len={actual_max_seq_len}")
    else:
        actual_vocab_size = vocab_size
        actual_d_model = 256
        actual_max_seq_len = 512
    
    # Create model with same config
    config = checkpoint.get('config', {})
    
    # Handle both dict and TrainingConfig object
    if hasattr(config, 'model'):
        model_config = config.model
    elif isinstance(config, dict):
        model_config = config
    else:
        model_config = {}
    
    # Extract model params (use detected values as fallback)
    d_model = getattr(model_config, 'd_model', actual_d_model) if hasattr(model_config, 'd_model') else model_config.get('d_model', actual_d_model)
    n_heads = getattr(model_config, 'n_heads', 4) if hasattr(model_config, 'n_heads') else model_config.get('n_heads', 4)
    n_layers = getattr(model_config, 'n_layers', 4) if hasattr(model_config, 'n_layers') else model_config.get('n_layers', 4)
    d_ff = getattr(model_config, 'd_ff', 1024) if hasattr(model_config, 'd_ff') else model_config.get('d_ff', 1024)
    max_seq_len = getattr(model_config, 'max_seq_length', actual_max_seq_len) if hasattr(model_config, 'max_seq_length') else model_config.get('max_seq_length', actual_max_seq_len)
    dropout = getattr(model_config, 'dropout', 0.1) if hasattr(model_config, 'dropout') else model_config.get('dropout', 0.1)
    
    # Use actual vocab size from checkpoint to avoid mismatch
    vocab_size = actual_vocab_size
    
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8, top_k=50):
    """Generate text from a prompt."""
    device = next(model.parameters()).device
    
    # Get model's max sequence length
    max_seq_len = model.pos_embedding.weight.shape[0] if hasattr(model, 'pos_embedding') else 512
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Truncate to max sequence length if needed
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]
            
            # Get predictions
            outputs = model(input_ids)
            logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for end of sequence (if tokenizer has special tokens)
            pad_id = getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'pad_id', None)
            if pad_id is not None and next_token.item() == pad_id:
                break
    
    # Decode
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def interactive_mode(model, tokenizer, temperature=0.8, max_length=200):
    """Run in interactive mode."""
    print("\n" + "="*60)
    print("🤖 Nima Interactive Mode")
    print("="*60)
    print("Ask me anything! Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not prompt:
                continue
            
            print("\nNima: ", end="", flush=True)
            response = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=max_length,
                temperature=temperature
            )
            
            # Extract just the response (after the prompt)
            if prompt in response:
                response = response[len(prompt):].strip()
            
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ask questions to Nima')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    parser.add_argument('--prompt', help='Single question to ask')
    parser.add_argument('--max-length', type=int, default=200, help='Maximum response length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    
    # Determine tokenizer type from path
    if 'bpe' in args.tokenizer.lower():
        tokenizer = SimpleBPETokenizer()
    elif 'word' in args.tokenizer.lower():
        tokenizer = WordTokenizer()
    else:
        tokenizer = CharTokenizer()
    
    tokenizer.load(args.tokenizer)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load model
    model = load_model(args.checkpoint, tokenizer)
    
    if args.interactive or not args.prompt:
        # Interactive mode
        interactive_mode(model, tokenizer, args.temperature, args.max_length)
    else:
        # Single question mode
        print(f"\nQuestion: {args.prompt}")
        print("\nNima: ", end="", flush=True)
        response = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Extract response after prompt
        if args.prompt in response:
            response = response[len(args.prompt):].strip()
        
        print(response)
        print()


if __name__ == '__main__':
    main()
