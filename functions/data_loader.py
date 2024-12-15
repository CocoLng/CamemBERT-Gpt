import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from transformers import RobertaTokenizerFast
import os
import psutil
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class DatasetConfig:
    """Configuration for dataset loading with optimized buffer handling"""
    name: str = "oscar"
    subset: Optional[str] = "unshuffled_deduplicated_fr"
    split: str = "train"
    streaming: bool = True
    verification_mode: str = "no_checks"
    
    @staticmethod
    def calculate_optimal_buffer(target_size_gb: float) -> int:
        """Calculate optimal buffer size based on available memory and dataset size"""
        try:
            mem_available = psutil.virtual_memory().available / (1024**3)
            
            if target_size_gb < 5:
                base_buffer = 25000
            elif target_size_gb < 20:
                base_buffer = 50000
            else:
                base_buffer = 100000
                
            mem_factor = min(mem_available * 0.15 / target_size_gb, 1.0)
            return int(base_buffer * mem_factor)
        except Exception as e:
            logging.warning(f"Buffer size calculation failed: {e}, using default")
            return 10000

class DataLoader:
    """Handles dataset loading and preprocessing for CamemBERT training"""
    
    def __init__(
        self,
        dataset_config: Optional[DatasetConfig] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = self._initialize_tokenizer()
        self.dataset = None
        self._dataset_size = 0
        self.dataset_config = dataset_config or DatasetConfig()

    def _initialize_tokenizer(self) -> RobertaTokenizerFast:
        """Initialize RoBERTa tokenizer"""
        try:
            return RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def _tokenize_function(self, examples: Dict) -> Dict:
        """RoBERTa-specific tokenization with type verification"""
        try:
            if not isinstance(examples["text"], (str, list)):
                self.logger.warning(f"Unexpected text type: {type(examples['text'])}")
                if isinstance(examples["text"], dict) and "text" in examples["text"]:
                    examples["text"] = examples["text"]["text"]
            
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_special_tokens_mask=True,
                return_attention_mask=True
            )
            
            return {k: list(v) if isinstance(v, (list, np.ndarray)) else v 
                for k, v in tokenized.items()}
                
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}", exc_info=True)
            raise

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset with proper text extraction and tokenization"""
        try:
            # Extraire le texte et tokenizer
            dataset = dataset.map(self._extract_text)
            dataset = dataset.select_columns(['text'])
            dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=['text']
            )
            return dataset
            
        except Exception as e:
            self.logger.error(f"Dataset preparation error: {e}")
            raise

    @staticmethod
    def _extract_text(example: Dict) -> Dict:
        """Extract text content from various dataset formats"""
        try:
            if 'text' in example:
                if isinstance(example['text'], list):
                    texts = [item['text'] if isinstance(item, dict) else item 
                            for item in example['text']]
                    return {'text': ' '.join(texts)}
                elif isinstance(example['text'], dict):
                    return {'text': example['text'].get('text', '')}
                elif isinstance(example['text'], str):
                    return {'text': example['text']}
            return {'text': ''}
        except Exception as e:
            logging.error(f"Text extraction error: {e}")
            return {'text': ''}

    def load_dataset(self, size_gb: float) -> bool:
        """Load dataset with specified size in GB"""
        try:
            # Calculate dataset size in tokens
            bytes_per_token = 4
            tokens_per_gb = int((1024 * 1024 * 1024) / bytes_per_token)
            self._dataset_size = int(size_gb * tokens_per_gb)
            
            # Calculate optimal buffer
            optimal_buffer = DatasetConfig.calculate_optimal_buffer(size_gb)
            
            # Load streaming dataset
            self.dataset = load_dataset(
                self.dataset_config.name,
                name=self.dataset_config.subset,
                split=self.dataset_config.split,
                streaming=True,
                trust_remote_code=True
            )
            
            if not self.dataset:
                raise ValueError(f"Failed to load dataset {self.dataset_config.name}")

            # Configure streaming pipeline
            self.dataset = self.dataset.shuffle(buffer_size=optimal_buffer)
            self.dataset = self.prepare_dataset(self.dataset)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset loading error: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if dataset is properly loaded and ready for use"""
        if not self.dataset:
            self.logger.warning("Dataset not loaded")
            return False
                
        if not self._dataset_size > 0:
            self.logger.warning("Dataset size is 0")
            return False
                
        try:
            next(iter(self.dataset))
            return True
        except Exception as e:
            self.logger.warning(f"Dataset not properly initialized: {e}")
            return False

    @property
    def dataset_size(self) -> int:
        """Get current dataset size in tokens"""
        return self._dataset_size