"""
Data preprocessing utilities.

This module provides functions for text cleaning, preprocessing,
and data augmentation for language model training.
"""

import re
import unicodedata
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import json


class TextPreprocessor:
    """
    Text preprocessing pipeline for cleaning and normalizing text data.
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phone_numbers: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
        min_length: int = 10,
        max_length: Optional[int] = None,
        custom_filters: Optional[List[Callable[[str], str]]] = None
    ):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_phone_numbers: Remove phone numbers
            normalize_whitespace: Normalize whitespace characters
            normalize_unicode: Normalize unicode characters
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
            custom_filters: List of custom filter functions
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length
        self.max_length = max_length
        self.custom_filters = custom_filters or []
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def process_text(self, text: str) -> str:
        """
        Process a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        if not isinstance(text, str):
            return ""
            
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Remove phone numbers
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub(' ', text)
            
        # Apply custom filters
        for filter_func in self.custom_filters:
            text = filter_func(text)
            
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
            text = text.strip()
            
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
            
        return text
        
    def process_texts(self, texts: List[str]) -> List[str]:
        """
        Process a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed texts
        """
        processed_texts = []
        
        for text in texts:
            processed_text = self.process_text(text)
            
            # Apply length filters
            if len(processed_text) < self.min_length:
                continue
                
            if self.max_length and len(processed_text) > self.max_length:
                processed_text = processed_text[:self.max_length]
                
            processed_texts.append(processed_text)
            
        return processed_texts
        
    def save_config(self, config_path: str) -> None:
        """Save preprocessor configuration."""
        config = {
            'lowercase': self.lowercase,
            'remove_urls': self.remove_urls,
            'remove_emails': self.remove_emails,
            'remove_phone_numbers': self.remove_phone_numbers,
            'normalize_whitespace': self.normalize_whitespace,
            'normalize_unicode': self.normalize_unicode,
            'min_length': self.min_length,
            'max_length': self.max_length
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_config(cls, config_path: str) -> 'TextPreprocessor':
        """Load preprocessor from configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return cls(**config)


class TextAugmenter:
    """
    Text augmentation utilities for data augmentation.
    """
    
    @staticmethod
    def add_noise(text: str, noise_prob: float = 0.1) -> str:
        """
        Add character-level noise to text.
        
        Args:
            text: Input text
            noise_prob: Probability of adding noise to each character
            
        Returns:
            Text with added noise
        """
        import random
        
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < noise_prob:
                # Randomly replace, insert, or delete character
                action = random.choice(['replace', 'insert', 'delete'])
                
                if action == 'replace' and chars[i].isalpha():
                    chars[i] = chr(ord('a') + random.randint(0, 25))
                elif action == 'insert':
                    chars.insert(i, chr(ord('a') + random.randint(0, 25)))
                elif action == 'delete' and len(chars) > 1:
                    chars.pop(i)
                    
        return ''.join(chars)
        
    @staticmethod
    def shuffle_sentences(text: str, shuffle_prob: float = 0.1) -> str:
        """
        Randomly shuffle sentences in text.
        
        Args:
            text: Input text
            shuffle_prob: Probability of shuffling sentences
            
        Returns:
            Text with shuffled sentences
        """
        import random
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if random.random() < shuffle_prob and len(sentences) > 1:
            random.shuffle(sentences)
            
        return '. '.join(sentences) + '.'
        
    @staticmethod
    def mask_tokens(text: str, mask_prob: float = 0.15, mask_token: str = '<mask>') -> str:
        """
        Randomly mask tokens in text (for BERT-style training).
        
        Args:
            text: Input text
            mask_prob: Probability of masking each token
            mask_token: Token to use for masking
            
        Returns:
            Text with masked tokens
        """
        import random
        
        words = text.split()
        masked_words = []
        
        for word in words:
            if random.random() < mask_prob:
                masked_words.append(mask_token)
            else:
                masked_words.append(word)
                
        return ' '.join(masked_words)


def clean_text_file(
    input_path: str,
    output_path: str,
    preprocessor: Optional[TextPreprocessor] = None,
    chunk_size: int = 1000
) -> Dict[str, Any]:
    """
    Clean a large text file in chunks.
    
    Args:
        input_path: Path to input text file
        output_path: Path to output cleaned file
        preprocessor: Text preprocessor instance
        chunk_size: Number of lines to process at once
        
    Returns:
        Dictionary with processing statistics
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
        
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    processed_lines = 0
    total_chars_before = 0
    total_chars_after = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            chunk = []
            
            for line in infile:
                total_lines += 1
                total_chars_before += len(line)
                chunk.append(line.strip())
                
                if len(chunk) >= chunk_size:
                    processed_chunk = preprocessor.process_texts(chunk)
                    for processed_line in processed_chunk:
                        if processed_line:  # Skip empty lines
                            outfile.write(processed_line + '\n')
                            processed_lines += 1
                            total_chars_after += len(processed_line) + 1
                    chunk = []
                    
            # Process remaining lines
            if chunk:
                processed_chunk = preprocessor.process_texts(chunk)
                for processed_line in processed_chunk:
                    if processed_line:
                        outfile.write(processed_line + '\n')
                        processed_lines += 1
                        total_chars_after += len(processed_line) + 1
                        
    stats = {
        'input_file': str(input_path),
        'output_file': str(output_path),
        'total_lines': total_lines,
        'processed_lines': processed_lines,
        'lines_filtered': total_lines - processed_lines,
        'chars_before': total_chars_before,
        'chars_after': total_chars_after,
        'compression_ratio': total_chars_after / total_chars_before if total_chars_before > 0 else 0
    }
    
    return stats


def split_text_file(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle_lines: bool = True
) -> Dict[str, str]:
    """
    Split a text file into train/validation/test sets.
    
    Args:
        input_path: Path to input text file
        output_dir: Directory to save split files
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        shuffle_lines: Whether to shuffle lines before splitting
        
    Returns:
        Dictionary with paths to split files
    """
    import random
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    # Shuffle if requested
    if shuffle_lines:
        random.shuffle(lines)
        
    # Calculate split indices
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)
    
    # Split data
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    # Save splits
    splits = {}
    
    if train_lines:
        train_path = output_dir / 'train.txt'
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_lines) + '\n')
        splits['train'] = str(train_path)
        
    if val_lines:
        val_path = output_dir / 'val.txt'
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_lines) + '\n')
        splits['val'] = str(val_path)
        
    if test_lines:
        test_path = output_dir / 'test.txt'
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_lines) + '\n')
        splits['test'] = str(test_path)
        
    print(f"Split {total_lines} lines:")
    print(f"  Train: {len(train_lines)} lines")
    print(f"  Validation: {len(val_lines)} lines") 
    print(f"  Test: {len(test_lines)} lines")
    
    return splits


def create_preprocessing_pipeline(
    input_path: str,
    output_dir: str,
    preprocessor_config: Optional[Dict] = None,
    split_ratios: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create a complete preprocessing pipeline.
    
    Args:
        input_path: Path to raw text file
        output_dir: Directory for processed files
        preprocessor_config: Configuration for text preprocessor
        split_ratios: Train/val/test split ratios
        
    Returns:
        Dictionary with pipeline results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configurations
    if preprocessor_config is None:
        preprocessor_config = {
            'normalize_whitespace': True,
            'normalize_unicode': True,
            'min_length': 50,
            'max_length': 10000
        }
        
    if split_ratios is None:
        split_ratios = {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1}
        
    # Create preprocessor
    preprocessor = TextPreprocessor(**preprocessor_config)
    
    # Clean text
    cleaned_path = output_dir / 'cleaned.txt'
    clean_stats = clean_text_file(input_path, cleaned_path, preprocessor)
    
    # Split data
    split_paths = split_text_file(cleaned_path, output_dir, **split_ratios)
    
    # Save configurations
    config_path = output_dir / 'preprocessing_config.json'
    preprocessor.save_config(config_path)
    
    results = {
        'clean_stats': clean_stats,
        'split_paths': split_paths,
        'config_path': str(config_path),
        'preprocessor_config': preprocessor_config,
        'split_ratios': split_ratios
    }
    
    return results
