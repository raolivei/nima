"""
Dataset loading and preprocessing utilities.

This module provides classes for loading and processing text datasets
for language model training.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Iterator
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
import tarfile

from .tokenizer import create_tokenizer


class TextDataset(Dataset):
    """
    Dataset class for text data with tokenization.
    
    Handles tokenization, sequence length management, and batching.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = None,
        return_tensors: bool = True
    ):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences (defaults to max_length)
            return_tensors: Whether to return PyTorch tensors
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        self.return_tensors = return_tensors
        
        # Tokenize all texts and create sequences
        self.sequences = []
        self._prepare_sequences()
        
    def _prepare_sequences(self):
        """Tokenize texts and create sequences of specified length."""
        for text in self.texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Create overlapping sequences
            for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                sequence = token_ids[i:i + self.max_length]
                if len(sequence) == self.max_length:
                    self.sequences.append(sequence)
                    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Get a sequence by index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary with input_ids and labels
        """
        sequence = self.sequences[idx]
        
        # For language modeling, input is sequence[:-1] and target is sequence[1:]
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        if self.return_tensors:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:
            return {
                'input_ids': input_ids,
                'labels': labels
            }


class CausalLMDataset(Dataset):
    """
    Dataset for causal (autoregressive) language modeling.
    
    Optimized for GPT-style training where the model predicts the next token.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        block_size: int = 1024,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize causal LM dataset.
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer instance
            block_size: Size of each training block
            cache_dir: Directory to cache tokenized data
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Check for cached data
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{self.file_path.stem}_tokenized.pt"
            
        if cache_file and cache_file.exists():
            print(f"Loading cached tokenized data from {cache_file}")
            self.token_ids = torch.load(cache_file)
        else:
            print(f"Tokenizing {self.file_path}")
            self.token_ids = self._tokenize_file()
            
            if cache_file:
                print(f"Caching tokenized data to {cache_file}")
                torch.save(self.token_ids, cache_file)
                
        print(f"Dataset size: {len(self.token_ids)} tokens, {len(self)} blocks")
        
    def _tokenize_file(self) -> torch.Tensor:
        """Tokenize the entire file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(token_ids, dtype=torch.long)
        
    def __len__(self) -> int:
        """Return number of blocks."""
        return len(self.token_ids) // self.block_size
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a block by index.
        
        Args:
            idx: Block index
            
        Returns:
            Dictionary with input_ids and labels
        """
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1  # +1 for label
        
        block = self.token_ids[start_idx:end_idx]
        
        return {
            'input_ids': block[:-1],
            'labels': block[1:]
        }


class DatasetDownloader:
    """
    Utility class for downloading common text datasets.
    """
    
    DATASETS = {
        'tiny_shakespeare': {
            'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
            'filename': 'tiny_shakespeare.txt'
        },
        'wikitext2': {
            'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
            'filename': 'wikitext-2-v1.zip',
            'extract': True
        },
        'gutenberg': {
            'url': 'https://www.gutenberg.org/files/74/74-0.txt',
            'filename': 'adventures_of_tom_sawyer.txt'
        }
    }
    
    @staticmethod
    def download_dataset(dataset_name: str, data_dir: str = 'data/raw') -> str:
        """
        Download a dataset.
        
        Args:
            dataset_name: Name of dataset to download
            data_dir: Directory to save dataset
            
        Returns:
            Path to downloaded/extracted dataset file
            
        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in DatasetDownloader.DATASETS:
            available = list(DatasetDownloader.DATASETS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not available. Available: {available}")
            
        dataset_info = DatasetDownloader.DATASETS[dataset_name]
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = data_dir / dataset_info['filename']
        
        # Download if not exists
        if not file_path.exists():
            print(f"Downloading {dataset_name} dataset...")
            response = requests.get(dataset_info['url'])
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded to {file_path}")
            
        # Extract if needed
        if dataset_info.get('extract', False):
            extract_dir = data_dir / dataset_name
            extract_dir.mkdir(exist_ok=True)
            
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif file_path.suffix in ['.tar', '.gz']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    
            # Find the main text file
            text_files = list(extract_dir.glob('**/*.txt'))
            if text_files:
                return str(text_files[0])
            else:
                return str(extract_dir)
        
        return str(file_path)


class DataCollator:
    """
    Data collator for batching sequences with padding.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer instance (must have pad_token_id)
            pad_to_multiple_of: Pad sequences to multiple of this value
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = getattr(tokenizer, 'vocab', {}).get('<pad>', 0)
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched and padded tensors
        """
        # Get maximum length in batch
        max_length = max(len(sample['input_ids']) for sample in batch)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad sequences
        input_ids = []
        labels = []
        attention_masks = []
        
        for sample in batch:
            input_seq = sample['input_ids']
            label_seq = sample['labels']
            
            # Calculate padding needed
            padding_length = max_length - len(input_seq)
            
            # Pad input_ids and labels
            padded_input = torch.cat([
                input_seq,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            
            padded_labels = torch.cat([
                label_seq,
                torch.full((padding_length,), -100, dtype=torch.long)  # -100 is ignored in loss
            ])
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat([
                torch.ones(len(input_seq), dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long)
            ])
            
            input_ids.append(padded_input)
            labels.append(padded_labels)
            attention_masks.append(attention_mask)
            
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_masks)
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_fn: Custom collate function
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )


def prepare_dataset(
    dataset_name: str,
    tokenizer_type: str = 'char',
    data_dir: str = 'data',
    max_length: int = 512,
    vocab_size: int = 10000,
    train_split: float = 0.9
) -> Tuple[Dataset, Dataset, object]:
    """
    Prepare a dataset for training.
    
    Args:
        dataset_name: Name of dataset to prepare
        tokenizer_type: Type of tokenizer to use
        data_dir: Directory for data storage
        max_length: Maximum sequence length
        vocab_size: Vocabulary size for tokenizer
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
    """
    # Download dataset
    data_path = DatasetDownloader.download_dataset(dataset_name, f"{data_dir}/raw")
    
    # Load text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into train and validation
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create tokenizer
    tokenizer_kwargs = {}
    if tokenizer_type in ['word', 'bpe'] and vocab_size is not None:
        tokenizer_kwargs['vocab_size'] = vocab_size
    
    tokenizer = create_tokenizer(tokenizer_type, **tokenizer_kwargs)
    
    # Build vocabulary on training text
    tokenizer.build_vocab([train_text])
    
    # Save tokenizer
    tokenizer_path = Path(data_dir) / 'processed' / f'{dataset_name}_{tokenizer_type}_tokenizer.json'
    tokenizer.save(str(tokenizer_path))
    
    # Create datasets
    train_dataset = TextDataset([train_text], tokenizer, max_length)
    val_dataset = TextDataset([val_text], tokenizer, max_length)
    
    print(f"Dataset prepared:")
    print(f"  Tokenizer: {tokenizer_type}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Training sequences: {len(train_dataset)}")
    print(f"  Validation sequences: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer
