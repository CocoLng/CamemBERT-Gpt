from dataclasses import dataclass
from transformers import RobertaConfig
import torch
import logging
from typing import Optional, Dict
import json
import os

@dataclass
class ModelArguments:
    vocab_size: int = 50265  # Default RoBERTa vocab size
    max_position_embeddings: int = 514  # Default + 2 for special tokens
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    
class ModelConfig:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.model_args = ModelArguments()
        
    def initialize_config(self, **kwargs) -> None:
        """Initialize RoBERTa configuration with custom parameters"""
        try:
            # Update model arguments with provided kwargs
            for key, value in kwargs.items():
                if hasattr(self.model_args, key):
                    setattr(self.model_args, key, value)
            
            # Create RoBERTa config
            self.config = RobertaConfig(
                vocab_size=self.model_args.vocab_size,
                max_position_embeddings=self.model_args.max_position_embeddings,
                hidden_size=self.model_args.hidden_size,
                num_attention_heads=self.model_args.num_attention_heads,
                num_hidden_layers=self.model_args.num_hidden_layers,
                intermediate_size=self.model_args.intermediate_size,
                hidden_act=self.model_args.hidden_act,
                hidden_dropout_prob=self.model_args.hidden_dropout_prob,
                attention_probs_dropout_prob=self.model_args.attention_probs_dropout_prob,
                type_vocab_size=self.model_args.type_vocab_size,
                layer_norm_eps=self.model_args.layer_norm_eps,
            )
            self.logger.info("Model configuration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model configuration: {e}")
            raise
            
    def validate_config(self) -> bool:
        """Validate the model configuration"""
        if not self.config:
            self.logger.error("Configuration not initialized")
            return False
            
        try:
            # Basic validation checks
            if self.config.hidden_size % self.config.num_attention_heads != 0:
                self.logger.error("Hidden size must be divisible by number of attention heads")
                return False
                
            if self.config.max_position_embeddings < 512:
                self.logger.warning("Max position embeddings less than RoBERTa default (512)")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False
            
    def save_config(self, path: str) -> None:
        """Save model configuration to disk"""
        if not self.config:
            raise ValueError("Configuration not initialized")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save configuration
            self.config.save_pretrained(path)
            self.logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
            
    def load_config(self, path: str) -> None:
        """Load model configuration from disk"""
        try:
            self.config = RobertaConfig.from_pretrained(path)
            self.logger.info(f"Configuration loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
            
    def get_config(self) -> Optional[RobertaConfig]:
        """Get the current model configuration"""
        return self.config
        
    def update_config(self, **kwargs) -> None:
        """Update specific configuration parameters"""
        if not self.config:
            raise ValueError("Configuration not initialized")
            
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Updated {key} to {value}")
                else:
                    self.logger.warning(f"Unknown configuration parameter: {key}")
                    
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise