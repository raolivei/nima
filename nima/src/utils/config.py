"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig


class Config:
    """Configuration manager for the LLM project."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = self.get_default_config()
    
    @staticmethod
    def load_config(config_path: str) -> DictConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return OmegaConf.create(config_dict)
    
    @staticmethod
    def get_default_config() -> DictConfig:
        """Get default configuration.
        
        Returns:
            Default configuration
        """
        default_config = {
            'model': {
                'vocab_size': 10000,
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 2048,
                'max_seq_len': 1024,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.0001,
                'num_epochs': 100,
                'warmup_steps': 4000,
                'weight_decay': 0.01,
                'gradient_clip': 1.0,
                'save_every': 1000,
                'eval_every': 500
            },
            'data': {
                'dataset': 'tiny_shakespeare',
                'train_split': 0.9,
                'val_split': 0.1,
                'tokenizer_type': 'char_level'
            },
            'paths': {
                'data_dir': 'data',
                'checkpoint_dir': 'experiments/checkpoints',
                'log_dir': 'experiments/logs'
            }
        }
        
        return OmegaConf.create(default_config)
    
    def save_config(self, save_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(self.config, f)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self.config = OmegaConf.merge(self.config, updates)
    
    def __getattr__(self, name: str) -> Any:
        """Access configuration attributes directly.
        
        Args:
            name: Attribute name
            
        Returns:
            Configuration value
        """
        return getattr(self.config, name)
    
    def __getitem__(self, key: str) -> Any:
        """Access configuration with dictionary syntax.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self.config[key]
