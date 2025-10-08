"""
Data processing and loading utilities.

This package provides comprehensive data processing capabilities for LLM training:
- Tokenizers (character, word, BPE)
- Dataset classes for text data
- Data preprocessing and cleaning
- Dataset downloading utilities
"""

# Tokenizers
from .tokenizer import (
    CharTokenizer,
    WordTokenizer,
    SimpleBPETokenizer,
    create_tokenizer
)

# Datasets and data loading
from .dataset import (
    TextDataset,
    CausalLMDataset,
    DatasetDownloader,
    DataCollator,
    create_dataloader,
    prepare_dataset
)

# Preprocessing utilities
from .preprocessing import (
    TextPreprocessor,
    TextAugmenter,
    clean_text_file,
    split_text_file,
    create_preprocessing_pipeline
)

__all__ = [
    # Tokenizers
    'CharTokenizer',
    'WordTokenizer',
    'SimpleBPETokenizer',
    'create_tokenizer',
    # Datasets
    'TextDataset',
    'CausalLMDataset',
    'DatasetDownloader',
    'DataCollator',
    'create_dataloader',
    'prepare_dataset',
    # Preprocessing
    'TextPreprocessor',
    'TextAugmenter',
    'clean_text_file',
    'split_text_file',
    'create_preprocessing_pipeline'
]
