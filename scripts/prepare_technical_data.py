#!/usr/bin/env python3
"""
Technical Data Preparation Script for Nima.

This script prepares technical documentation, engineering content, and Q&A data
for training Nima with a focus on system engineering knowledge.

Supports multiple data sources:
- Technical documentation (Markdown, RST, plain text)
- GitHub READMEs and documentation
- Q&A datasets (JSON, JSONL)
- Custom curated notes and transcripts

Features:
- Intelligent text cleaning and preprocessing
- Code block preservation
- Technical term recognition
- 80/10/10 train/val/test split
- Multiple tokenizer support
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import create_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source."""
    path: str
    type: str  # 'text', 'markdown', 'json', 'jsonl'
    format: Optional[str] = None  # For structured data: 'qa', 'chat', 'doc'
    weight: float = 1.0  # Relative weight for sampling


class TechnicalDataPreprocessor:
    """
    Preprocessor for technical documentation and code.
    
    Preserves:
    - Code blocks
    - Technical terms
    - Command-line examples
    - Configuration snippets
    """
    
    def __init__(self, preserve_code: bool = True, preserve_formatting: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            preserve_code: Keep code blocks intact
            preserve_formatting: Maintain technical formatting
        """
        self.preserve_code = preserve_code
        self.preserve_formatting = preserve_formatting
        
        # Technical patterns to preserve
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.inline_code_pattern = re.compile(r'`[^`]+`')
        self.command_pattern = re.compile(r'^\s*[\$#>]\s+.*$', re.MULTILINE)
        
    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving technical content.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Store code blocks
        code_blocks = []
        if self.preserve_code:
            for match in self.code_block_pattern.finditer(text):
                code_blocks.append(match.group(0))
                text = text.replace(match.group(0), f'__CODE_BLOCK_{len(code_blocks)-1}__')
        
        # Remove excessive whitespace but preserve formatting
        if not self.preserve_formatting:
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 newlines
            text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        
        # Remove common markdown artifacts (but keep structure)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # Headers
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
        
        # Restore code blocks
        for i, block in enumerate(code_blocks):
            text = text.replace(f'__CODE_BLOCK_{i}__', block)
        
        # Remove excessive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_from_markdown(self, content: str) -> List[str]:
        """
        Extract sections from markdown documentation.
        
        Args:
            content: Markdown content
            
        Returns:
            List of text sections
        """
        sections = []
        
        # Split by headers
        header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        parts = header_pattern.split(content)
        
        # Combine header with content
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                header = parts[i].strip()
                body = parts[i + 1].strip()
                if body:
                    sections.append(f"# {header}\n\n{body}")
        
        return sections if sections else [content]


class TechnicalDatasetBuilder:
    """
    Build training datasets from multiple technical sources.
    """
    
    def __init__(
        self,
        output_dir: str,
        tokenizer_type: str = 'bpe',
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_length: int = 512,
        min_length: int = 50
    ):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory for processed data
            tokenizer_type: Type of tokenizer ('char', 'word', 'bpe')
            split_ratios: (train, val, test) ratios
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer_type = tokenizer_type
        self.split_ratios = split_ratios
        self.max_length = max_length
        self.min_length = min_length
        
        self.preprocessor = TechnicalDataPreprocessor()
        self.tokenizer = None
        
        logger.info(f"Initialized dataset builder: {tokenizer_type} tokenizer")
        logger.info(f"Split ratios: train={split_ratios[0]}, val={split_ratios[1]}, test={split_ratios[2]}")
    
    def load_text_file(self, filepath: str) -> str:
        """Load text from file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def load_json_dataset(self, filepath: str, format_type: str = 'qa') -> List[str]:
        """
        Load structured JSON dataset.
        
        Args:
            filepath: Path to JSON file
            format_type: Format ('qa', 'chat', 'doc')
            
        Returns:
            List of formatted text samples
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        if format_type == 'qa':
            # Q&A format: {"question": "...", "answer": "..."}
            for item in data:
                if 'question' in item and 'answer' in item:
                    samples.append(f"Q: {item['question']}\n\nA: {item['answer']}")
        
        elif format_type == 'chat':
            # Chat format: {"messages": [{"role": "...", "content": "..."}]}
            for item in data:
                if 'messages' in item:
                    text = []
                    for msg in item['messages']:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        text.append(f"{role.capitalize()}: {content}")
                    samples.append('\n\n'.join(text))
        
        elif format_type == 'doc':
            # Document format: {"title": "...", "content": "..."}
            for item in data:
                title = item.get('title', '')
                content = item.get('content', '')
                if title and content:
                    samples.append(f"# {title}\n\n{content}")
                elif content:
                    samples.append(content)
        
        return samples
    
    def load_jsonl_dataset(self, filepath: str, format_type: str = 'qa') -> List[str]:
        """Load JSONL dataset (one JSON object per line)."""
        samples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    if format_type == 'qa' and 'question' in item and 'answer' in item:
                        samples.append(f"Q: {item['question']}\n\nA: {item['answer']}")
                    elif format_type == 'doc':
                        content = item.get('text', item.get('content', ''))
                        if content:
                            samples.append(content)
                
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
        
        return samples
    
    def process_data_source(self, source: DataSource) -> List[str]:
        """
        Process a single data source.
        
        Args:
            source: DataSource configuration
            
        Returns:
            List of processed text samples
        """
        logger.info(f"Processing {source.type} data from: {source.path}")
        
        samples = []
        
        if source.type == 'text':
            text = self.load_text_file(source.path)
            cleaned = self.preprocessor.clean_text(text)
            samples.append(cleaned)
        
        elif source.type == 'markdown':
            content = self.load_text_file(source.path)
            sections = self.preprocessor.extract_from_markdown(content)
            for section in sections:
                cleaned = self.preprocessor.clean_text(section)
                if len(cleaned) >= self.min_length:
                    samples.append(cleaned)
        
        elif source.type == 'json':
            samples = self.load_json_dataset(source.path, source.format or 'qa')
            samples = [self.preprocessor.clean_text(s) for s in samples]
        
        elif source.type == 'jsonl':
            samples = self.load_jsonl_dataset(source.path, source.format or 'qa')
            samples = [self.preprocessor.clean_text(s) for s in samples]
        
        # Filter by length
        samples = [s for s in samples if self.min_length <= len(s) <= self.max_length * 10]
        
        logger.info(f"Extracted {len(samples)} samples from {source.path}")
        return samples
    
    def build_dataset(self, sources: List[DataSource]) -> Dict[str, str]:
        """
        Build complete dataset from multiple sources.
        
        Args:
            sources: List of data sources
            
        Returns:
            Dictionary with paths to train/val/test files
        """
        # Collect all samples
        all_samples = []
        for source in sources:
            samples = self.process_data_source(source)
            # Apply weight
            weighted_samples = samples * int(source.weight)
            all_samples.extend(weighted_samples)
        
        logger.info(f"Total samples collected: {len(all_samples)}")
        
        # Shuffle samples
        import random
        random.seed(42)
        random.shuffle(all_samples)
        
        # Split data
        train_ratio, val_ratio, test_ratio = self.split_ratios
        n = len(all_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples = all_samples[:train_end]
        val_samples = all_samples[train_end:val_end]
        test_samples = all_samples[val_end:]
        
        logger.info(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        # Build tokenizer
        logger.info(f"Building {self.tokenizer_type} tokenizer...")
        self.tokenizer = create_tokenizer(self.tokenizer_type)
        
        # Combine all text for vocabulary
        all_text = '\n\n'.join(all_samples)
        self.tokenizer.build_vocab(all_text)
        
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
        
        # Save tokenizer
        tokenizer_path = self.output_dir / f'tokenizer_{self.tokenizer_type}.json'
        self.tokenizer.save(str(tokenizer_path))
        logger.info(f"Saved tokenizer to: {tokenizer_path}")
        
        # Save splits
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        paths = {}
        for split_name, samples in splits.items():
            split_path = self.output_dir / f'{split_name}.txt'
            with open(split_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(samples))
            paths[split_name] = str(split_path)
            logger.info(f"Saved {split_name} split to: {split_path}")
        
        # Save metadata
        metadata = {
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.tokenizer.vocab_size,
            'num_sources': len(sources),
            'split_ratios': self.split_ratios,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'max_length': self.max_length,
            'min_length': self.min_length
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")
        
        return paths


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description='Prepare technical datasets for Nima')
    
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--tokenizer', type=str, default='bpe',
                       choices=['char', 'word', 'bpe'],
                       help='Tokenizer type')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum sequence length')
    
    # Data sources
    parser.add_argument('--text-files', nargs='+', default=[],
                       help='Plain text or markdown files')
    parser.add_argument('--json-files', nargs='+', default=[],
                       help='JSON files (Q&A or doc format)')
    parser.add_argument('--jsonl-files', nargs='+', default=[],
                       help='JSONL files')
    parser.add_argument('--format', type=str, default='qa',
                       choices=['qa', 'chat', 'doc'],
                       help='Format for JSON/JSONL files')
    
    # Split configuration
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation data ratio')
    
    args = parser.parse_args()
    
    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    split_ratios = (args.train_ratio, args.val_ratio, test_ratio)
    
    # Build data sources
    sources = []
    
    for filepath in args.text_files:
        file_type = 'markdown' if filepath.endswith('.md') else 'text'
        sources.append(DataSource(filepath, file_type))
    
    for filepath in args.json_files:
        sources.append(DataSource(filepath, 'json', format=args.format))
    
    for filepath in args.jsonl_files:
        sources.append(DataSource(filepath, 'jsonl', format=args.format))
    
    if not sources:
        logger.error("No data sources specified!")
        logger.info("Use --text-files, --json-files, or --jsonl-files to provide data")
        return
    
    # Build dataset
    builder = TechnicalDatasetBuilder(
        output_dir=args.output_dir,
        tokenizer_type=args.tokenizer,
        split_ratios=split_ratios,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    paths = builder.build_dataset(sources)
    
    logger.info("=" * 80)
    logger.info("Dataset preparation complete!")
    logger.info("=" * 80)
    logger.info(f"Train data: {paths['train']}")
    logger.info(f"Val data: {paths['val']}")
    logger.info(f"Test data: {paths['test']}")
    logger.info(f"Tokenizer: {args.output_dir}/tokenizer_{args.tokenizer}.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
