#!/usr/bin/env python3
"""
Data preparation script for LLM training.

This script demonstrates how to:
1. Download and preprocess text datasets
2. Create tokenizers
3. Prepare training and validation datasets
4. Save processed data for training
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data import (
        DatasetDownloader,
        TextPreprocessor,
        create_tokenizer,
        prepare_dataset,
        create_preprocessing_pipeline
    )
    DATA_AVAILABLE = True
except ImportError as e:
    print(f"Data modules not available: {e}")
    DATA_AVAILABLE = False


def prepare_tiny_shakespeare(
    data_dir: str = 'data',
    tokenizer_type: str = 'char',
    vocab_size: int = 10000,
    max_length: int = 512
):
    """
    Prepare the Tiny Shakespeare dataset.
    
    Args:
        data_dir: Directory for data storage
        tokenizer_type: Type of tokenizer ('char', 'word', 'bpe')
        vocab_size: Vocabulary size for tokenizer
        max_length: Maximum sequence length
    """
    print("=" * 60)
    print("Preparing Tiny Shakespeare Dataset")
    print("=" * 60)
    
    # Download dataset
    print("Downloading dataset...")
    data_path = DatasetDownloader.download_dataset('tiny_shakespeare', f"{data_dir}/raw")
    print(f"Dataset downloaded to: {data_path}")
    
    # Preprocess text
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor(
        normalize_whitespace=True,
        normalize_unicode=True,
        min_length=10,
        max_length=50000  # Keep longer texts for Shakespeare
    )
    
    # Clean and split data
    processed_dir = Path(data_dir) / 'processed' / 'tiny_shakespeare'
    pipeline_results = create_preprocessing_pipeline(
        input_path=data_path,
        output_dir=str(processed_dir),
        preprocessor_config={
            'normalize_whitespace': True,
            'normalize_unicode': True,
            'min_length': 10
        },
        split_ratios={'train_ratio': 0.9, 'val_ratio': 0.1, 'test_ratio': 0.0}
    )
    
    print(f"Preprocessing complete:")
    for key, value in pipeline_results['clean_stats'].items():
        print(f"  {key}: {value}")
    
    # Create tokenizer
    print(f"\nCreating {tokenizer_type} tokenizer...")
    
    # Read training data for tokenizer
    train_path = pipeline_results['split_paths']['train']
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    # Create and train tokenizer
    if tokenizer_type == 'char':
        tokenizer = create_tokenizer('char')
    elif tokenizer_type == 'word':
        tokenizer = create_tokenizer('word', max_vocab_size=vocab_size)
    elif tokenizer_type == 'bpe':
        tokenizer = create_tokenizer('bpe', vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    tokenizer.build_vocab([train_text])
    
    # Save tokenizer
    tokenizer_path = processed_dir / f'tokenizer_{tokenizer_type}.json'
    tokenizer.save(str(tokenizer_path))
    
    print(f"Tokenizer created and saved:")
    print(f"  Type: {tokenizer_type}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Saved to: {tokenizer_path}")
    
    # Test tokenization
    print(f"\nTesting tokenization...")
    sample_text = "To be, or not to be, that is the question."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original: {sample_text}")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  {decoded}")
    
    return {
        'data_path': data_path,
        'processed_dir': processed_dir,
        'tokenizer_path': tokenizer_path,
        'tokenizer': tokenizer,
        'split_paths': pipeline_results['split_paths']
    }


def prepare_custom_dataset(
    file_path: str,
    dataset_name: str,
    data_dir: str = 'data',
    tokenizer_type: str = 'char',
    vocab_size: int = 10000
):
    """
    Prepare a custom dataset from a text file.
    
    Args:
        file_path: Path to input text file
        dataset_name: Name for the dataset
        data_dir: Directory for data storage
        tokenizer_type: Type of tokenizer
        vocab_size: Vocabulary size
    """
    print("=" * 60)
    print(f"Preparing Custom Dataset: {dataset_name}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    # Create preprocessing pipeline
    processed_dir = Path(data_dir) / 'processed' / dataset_name
    
    pipeline_results = create_preprocessing_pipeline(
        input_path=file_path,
        output_dir=str(processed_dir),
        preprocessor_config={
            'lowercase': False,  # Preserve case for literary texts
            'normalize_whitespace': True,
            'normalize_unicode': True,
            'remove_urls': True,
            'remove_emails': True,
            'min_length': 50
        }
    )
    
    print("Dataset preprocessing complete!")
    
    # Create tokenizer
    print(f"Creating {tokenizer_type} tokenizer...")
    
    train_path = pipeline_results['split_paths']['train']
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    tokenizer = create_tokenizer(
        tokenizer_type,
        vocab_size=vocab_size if tokenizer_type in ['word', 'bpe'] else None
    )
    tokenizer.build_vocab([train_text])
    
    # Save tokenizer
    tokenizer_path = processed_dir / f'tokenizer_{tokenizer_type}.json'
    tokenizer.save(str(tokenizer_path))
    
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return {
        'processed_dir': processed_dir,
        'tokenizer_path': tokenizer_path,
        'split_paths': pipeline_results['split_paths']
    }


def main():
    """Main function to run data preparation."""
    parser = argparse.ArgumentParser(description='Prepare datasets for LLM training')
    parser.add_argument('--dataset', default='tiny_shakespeare',
                       help='Dataset name (tiny_shakespeare, wikitext2, gutenberg) or path to custom file')
    parser.add_argument('--tokenizer', default='char',
                       choices=['char', 'word', 'bpe'],
                       help='Tokenizer type')
    parser.add_argument('--vocab_size', type=int, default=10000,
                       help='Vocabulary size (for word and BPE tokenizers)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--data_dir', default='data',
                       help='Data directory')
    
    args = parser.parse_args()
    
    if not DATA_AVAILABLE:
        print("Error: Data processing modules not available.")
        print("Please ensure all dependencies are installed.")
        return
    
    try:
        # Check if dataset is a built-in dataset or custom file
        builtin_datasets = ['tiny_shakespeare', 'wikitext2', 'gutenberg']
        
        if args.dataset in builtin_datasets:
            # Use built-in dataset preparation
            if args.dataset == 'tiny_shakespeare':
                results = prepare_tiny_shakespeare(
                    data_dir=args.data_dir,
                    tokenizer_type=args.tokenizer,
                    vocab_size=args.vocab_size,
                    max_length=args.max_length
                )
            else:
                # For other built-in datasets, use the general prepare_dataset function
                train_dataset, val_dataset, tokenizer = prepare_dataset(
                    dataset_name=args.dataset,
                    tokenizer_type=args.tokenizer,
                    data_dir=args.data_dir,
                    vocab_size=args.vocab_size,
                    max_length=args.max_length
                )
                results = {
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                    'tokenizer': tokenizer
                }
        else:
            # Treat as custom dataset file
            dataset_name = Path(args.dataset).stem
            results = prepare_custom_dataset(
                file_path=args.dataset,
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                tokenizer_type=args.tokenizer,
                vocab_size=args.vocab_size
            )
        
        if results:
            print("\n" + "=" * 60)
            print("Data preparation completed successfully!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Review the processed data files")
            print("2. Adjust tokenizer settings if needed")
            print("3. Start training with: python scripts/train.py")
            
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
