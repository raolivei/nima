"""
Tokenization utilities for text preprocessing.

This module provides different tokenization strategies:
- Character-level tokenization
- Word-level tokenization  
- Byte-Pair Encoding (BPE) tokenization
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter, defaultdict


class CharTokenizer:
    """
    Character-level tokenizer.
    
    Simple tokenization that treats each character as a token.
    Good for small datasets and demonstrating concepts.
    """
    
    def __init__(self, special_tokens: Optional[List[str]] = None):
        """
        Initialize character tokenizer.
        
        Args:
            special_tokens: List of special tokens to add
        """
        self.special_tokens = special_tokens or ['<pad>', '<unk>', '<sos>', '<eos>']
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
            
        # Count all characters
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)
            
        # Add characters to vocabulary
        current_id = len(self.special_tokens)
        for char, _ in char_counter.most_common():
            if char not in self.char_to_id:
                self.char_to_id[char] = current_id
                self.id_to_char[current_id] = char
                current_id += 1
                
        self.vocab_size = len(self.char_to_id)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add <sos> and <eos> tokens
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_special_tokens and '<sos>' in self.char_to_id:
            tokens.append(self.char_to_id['<sos>'])
            
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.char_to_id['<unk>'])
                
        if add_special_tokens and '<eos>' in self.char_to_id:
            tokens.append(self.char_to_id['<eos>'])
            
        return tokens
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                if skip_special_tokens and char in self.special_tokens:
                    continue
                chars.append(char)
                
        return ''.join(chars)
        
    def save(self, save_path: str) -> None:
        """Save tokenizer to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
    def load(self, load_path: str) -> None:
        """Load tokenizer from file."""
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        self.char_to_id = tokenizer_data['char_to_id']
        self.id_to_char = {int(k): v for k, v in tokenizer_data['id_to_char'].items()}
        self.special_tokens = tokenizer_data['special_tokens']
        self.vocab_size = tokenizer_data['vocab_size']


class WordTokenizer:
    """
    Word-level tokenizer with basic preprocessing.
    
    Splits text into words and handles punctuation.
    """
    
    def __init__(self, special_tokens: Optional[List[str]] = None, max_vocab_size: int = 10000):
        """
        Initialize word tokenizer.
        
        Args:
            special_tokens: List of special tokens to add
            max_vocab_size: Maximum vocabulary size
        """
        self.special_tokens = special_tokens or ['<pad>', '<unk>', '<sos>', '<eos>']
        self.max_vocab_size = max_vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by lowercasing and handling punctuation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Lowercase
        text = text.lower()
        
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
        # Count all words
        word_counter = Counter()
        for text in texts:
            processed_text = self._preprocess_text(text)
            words = processed_text.split()
            word_counter.update(words)
            
        # Add most common words to vocabulary
        current_id = len(self.special_tokens)
        vocab_limit = self.max_vocab_size - len(self.special_tokens)
        
        for word, _ in word_counter.most_common(vocab_limit):
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
                
        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add <sos> and <eos> tokens
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_special_tokens and '<sos>' in self.word_to_id:
            tokens.append(self.word_to_id['<sos>'])
            
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        for word in words:
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                tokens.append(self.word_to_id['<unk>'])
                
        if add_special_tokens and '<eos>' in self.word_to_id:
            tokens.append(self.word_to_id['<eos>'])
            
        return tokens
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if skip_special_tokens and word in self.special_tokens:
                    continue
                words.append(word)
                
        return ' '.join(words)
        
    def save(self, save_path: str) -> None:
        """Save tokenizer to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'special_tokens': self.special_tokens,
            'max_vocab_size': self.max_vocab_size,
            'vocab_size': self.vocab_size
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
    def load(self, load_path: str) -> None:
        """Load tokenizer from file."""
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        self.word_to_id = tokenizer_data['word_to_id']
        self.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
        self.special_tokens = tokenizer_data['special_tokens']
        self.max_vocab_size = tokenizer_data['max_vocab_size']
        self.vocab_size = tokenizer_data['vocab_size']


class SimpleBPETokenizer:
    """
    Simple Byte-Pair Encoding (BPE) tokenizer.
    
    Implements a basic version of BPE for subword tokenization.
    """
    
    def __init__(self, special_tokens: Optional[List[str]] = None, vocab_size: int = 8000):
        """
        Initialize BPE tokenizer.
        
        Args:
            special_tokens: List of special tokens to add
            vocab_size: Target vocabulary size
        """
        self.special_tokens = special_tokens or ['<pad>', '<unk>', '<sos>', '<eos>']
        self.target_vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.vocab_size = 0
        
    def _get_word_tokens(self, text: str) -> List[str]:
        """Split text into words and add end-of-word marker."""
        words = text.lower().split()
        return [word + '</w>' for word in words]
        
    def _get_pairs(self, word: str) -> set:
        """Get all adjacent character pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
        
    def build_vocab(self, texts: List[str], num_merges: Optional[int] = None) -> None:
        """
        Build BPE vocabulary from texts.
        
        Args:
            texts: List of text strings
            num_merges: Number of merge operations (defaults to target_vocab_size - base_vocab_size)
        """
        # Initialize vocabulary with characters
        vocab = {}
        
        # Process all texts to get word frequencies
        for text in texts:
            words = self._get_word_tokens(text)
            for word in words:
                vocab[' '.join(word)] = vocab.get(' '.join(word), 0) + 1
                
        # Get all characters as initial vocabulary
        chars = set()
        for word in vocab:
            chars.update(word.split())
            
        # Add special tokens and characters to vocabulary
        self.vocab = {}
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            
        for i, char in enumerate(sorted(chars)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                
        # Determine number of merges
        if num_merges is None:
            num_merges = max(0, self.target_vocab_size - len(self.vocab))
            
        # Perform BPE merges
        for _ in range(num_merges):
            pairs = defaultdict(int)
            
            # Count all pairs
            for word, freq in vocab.items():
                word_pairs = self._get_pairs(word.split())
                for pair in word_pairs:
                    pairs[pair] += freq
                    
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the best pair
            new_vocab = {}
            pattern = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            
            for word in vocab:
                new_word = word.replace(pattern, replacement)
                new_vocab[new_word] = vocab[word]
                
            vocab = new_vocab
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            if replacement not in self.vocab:
                self.vocab[replacement] = len(self.vocab)
                
        self.vocab_size = len(self.vocab)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs using BPE.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_special_tokens and '<sos>' in self.vocab:
            tokens.append(self.vocab['<sos>'])
            
        words = self._get_word_tokens(text)
        
        for word in words:
            # Start with character-level tokenization
            word_tokens = list(word)
            
            # Apply BPE merges
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == merge:
                        word_tokens = (word_tokens[:i] + 
                                     [''.join(merge)] + 
                                     word_tokens[i + 2:])
                    else:
                        i += 1
                        
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['<unk>'])
                    
        if add_special_tokens and '<eos>' in self.vocab:
            tokens.append(self.vocab['<eos>'])
            
        return tokens
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
                
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
        
    def save(self, save_path: str) -> None:
        """Save tokenizer to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'target_vocab_size': self.target_vocab_size,
            'vocab_size': self.vocab_size
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
    def load(self, load_path: str) -> None:
        """Load tokenizer from file."""
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        self.vocab = tokenizer_data['vocab']
        self.merges = [tuple(merge) for merge in tokenizer_data['merges']]
        self.special_tokens = tokenizer_data['special_tokens']
        self.target_vocab_size = tokenizer_data['target_vocab_size']
        self.vocab_size = tokenizer_data['vocab_size']


def create_tokenizer(tokenizer_type: str, **kwargs) -> Union[CharTokenizer, WordTokenizer, SimpleBPETokenizer]:
    """
    Create a tokenizer of the specified type.
    
    Args:
        tokenizer_type: Type of tokenizer ('char', 'word', 'bpe')
        **kwargs: Additional arguments for tokenizer initialization
        
    Returns:
        Initialized tokenizer
        
    Raises:
        ValueError: If tokenizer_type is not supported
    """
    if tokenizer_type == 'char':
        return CharTokenizer(**kwargs)
    elif tokenizer_type == 'word':
        return WordTokenizer(**kwargs)
    elif tokenizer_type == 'bpe':
        return SimpleBPETokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
