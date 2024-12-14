import logging
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast
from datasets import Dataset
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
            
            # Conservative buffer sizing for streaming
            if target_size_gb < 5:
                base_buffer = 25000
            elif target_size_gb < 20:
                base_buffer = 50000
            else:
                base_buffer = 100000
                
            # Use maximum 15% of available memory for buffer
            mem_factor = min(mem_available * 0.15 / target_size_gb, 1.0)
            return int(base_buffer * mem_factor)
        except Exception as e:
            logging.warning(f"Buffer size calculation failed: {e}, using default")
            return 10000

@dataclass
class MaskingResult:
    """Structure for masking results with clear metrics"""
    original_text: str
    masked_text: str
    masking_ratio: float
    num_masked_tokens: int
    total_tokens: int

@dataclass
class MaskingStats:
    """Structure for masking statistics"""
    average_ratio: float
    num_samples: int
    expected_ratio: float
    deviation: float

class DataLoader:
    def __init__(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        tokens_to_fuse: Optional[List[str]] = None,
        tokens_to_remove: Optional[List[str]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = self._initialize_tokenizer()
        self.dataset = None
        self._dataset_size = 0
        self.dataset_config = dataset_config or DatasetConfig()
        self.mlm_probability = 0.15  # Fixed as per RoBERTa
        self.data_collator = self._initialize_data_collator()
        
        # Token processing configuration
        self.tokens_to_fuse = tokens_to_fuse or []
        self.tokens_to_remove = tokens_to_remove or []

    def _initialize_data_collator(self) -> DataCollatorForLanguageModeling:
        """Initialize standard RoBERTa MLM collator"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=8 
        )
        

    def _initialize_tokenizer(self) -> RobertaTokenizerFast:
        """Initialize RoBERTa tokenizer with error handling"""
        try:
            return RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset with proper text extraction.
        """
        try:
            # Apply the transformation and keep only the text column
            dataset = dataset.map(extract_text)
            return dataset.select_columns(['text'])
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
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
            
            # Ensure all values are lists (not dicts)
            return {k: list(v) if isinstance(v, (list, np.ndarray)) else v 
                for k, v in tokenized.items()}
                
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}", exc_info=True)
            raise

    def load_streaming_dataset(self, size_gb: float) -> str:
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

            # Setup streaming pipeline with proper data extraction
            self.dataset = self.dataset.shuffle(buffer_size=optimal_buffer)
            # Apply text extraction
            self.dataset = self.dataset.map(
                extract_text
            )
            
            # Verify we have text before proceeding
            sample = next(iter(self.dataset))
            if 'text' not in sample or not sample['text']:
                raise ValueError("Failed to extract text from dataset")

            # Apply tokenization
            self.dataset = self.dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=['text']  # Remove text after tokenization
            )

            # Verify initial samples
            masking_stats = self._verify_streaming_masking()
            
            return self._format_loading_status(size_gb, masking_stats, optimal_buffer)
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            self.logger.error(error_msg, exc_info=True)  # Added exc_info for better debugging
            return f"❌ {error_msg}"
        
    def _verify_streaming_masking(self) -> MaskingStats:
        """Verify masking on streaming data with proper type handling"""
        try:
            total_ratio = 0
            samples_checked = 0
            max_samples = 5

            # Sample a few batches for verification
            iterator = iter(self.dataset)
            for _ in range(max_samples):
                try:
                    sample = next(iterator)
                    
                    # Debug logging
                    self.logger.debug(f"Sample keys: {sample.keys()}")
                    for k, v in sample.items():
                        self.logger.debug(f"Key: {k}, Type: {type(v)}")
                    
                    # Ensure all required keys are present and properly formatted
                    required_keys = {'input_ids', 'attention_mask', 'special_tokens_mask'}
                    if not all(k in sample for k in required_keys):
                        missing = required_keys - set(sample.keys())
                        raise ValueError(f"Missing required keys in sample: {missing}")
                    
                    # Convert to tensors with explicit type checking
                    sample_tensor = {}
                    for k, v in sample.items():
                        if isinstance(v, (list, np.ndarray)):
                            sample_tensor[k] = torch.tensor(v, dtype=torch.long)
                        elif isinstance(v, torch.Tensor):
                            sample_tensor[k] = v
                        else:
                            self.logger.warning(f"Unexpected type for key {k}: {type(v)}")
                            continue
                    
                    # Proceed only if we have valid tensors
                    if not all(k in sample_tensor for k in required_keys):
                        continue
                    
                    masked_batch = self.data_collator([sample_tensor])
                    
                    attention_mask = sample_tensor['attention_mask']
                    masked_tokens = (masked_batch['labels'][0] != -100).sum().item()
                    valid_tokens = attention_mask.sum().item()
                    
                    if valid_tokens > 0:
                        total_ratio += masked_tokens / valid_tokens
                        samples_checked += 1
                        
                except StopIteration:
                    break
                except Exception as e:
                    self.logger.warning(f"Error processing sample: {e}")
                    continue

            if samples_checked == 0:
                raise ValueError("No valid samples found for masking verification")

            avg_ratio = total_ratio / samples_checked
            
            return MaskingStats(
                average_ratio=avg_ratio,
                num_samples=samples_checked,
                expected_ratio=self.mlm_probability,
                deviation=abs(avg_ratio - self.mlm_probability)
            )
            
        except Exception as e:
            self.logger.error(f"Error in masking verification: {e}", exc_info=True)
            raise

        
    def _prepare_text(self, example: Dict) -> Dict:
        """Prepare text from streaming example"""
        if 'text' in example and isinstance(example['text'], list):
            texts = [item['text'] for item in example['text']]
            return {'text': ' '.join(texts)}
        elif 'content' in example:
            return {'text': example['content']}
        elif 'text' in example and isinstance(example['text'], str):
            return {'text': example['text']}
        return {'text': ''}

    def load_with_masking(self, size: float, prob: float) -> str:
        """Legacy method for compatibility with run_handler"""
        try:
            self.mlm_probability = prob
            self.data_collator = self._initialize_data_collator()
            return self.load_streaming_dataset(size)
        except Exception as e:
            error_msg = f"Error during loading: {e}"
            self.logger.error(error_msg)
            return error_msg

    def visualize_with_density(self, text: str, density: float) -> tuple[str, str]:
        """Visualize masking without dataset dependency"""
        try:
            # Si texte fourni, utiliser directement
            if text.strip():
                return self.visualize_masking(text)
            
            # Sinon prendre un exemple du stream
            sample = next(iter(self.dataset))
            text = sample['text']
            
            # Vérifier densité
            tokens = self.tokenizer(text, truncation=True, max_length=512)
            current_density = sum(tokens['attention_mask']) / len(tokens['attention_mask'])
            
            if current_density >= density:
                return self.visualize_masking(text)
            else:
                return "Text density too low", ""
                
        except Exception as e:
            return f"Error: {str(e)}", ""

    def visualize_masking(self, text: str) -> tuple[str, str]:
        """Visualize masking of input text"""
        try:
            if not text or not text.strip():
                return "❌ Texte vide", ""
                
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors="pt"
            )

            masked = self.data_collator([{
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0]
            }])

            original_text = self.tokenizer.decode(
                inputs["input_ids"][0],
                skip_special_tokens=True
            )

            masked_text = self.tokenizer.decode(
                masked["input_ids"][0],
                skip_special_tokens=False
            )
            
            return original_text, masked_text
        except Exception as e:
            error_msg = f"❌ Erreur lors du masquage: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, ""

    def _verify_masking(self) -> MaskingStats:
        """Verify masking on sample data"""
        try:
            samples = []
            for sample in self.dataset.take(5):
                sample = {
                    'input_ids': torch.tensor(sample['input_ids']),
                    'attention_mask': torch.tensor(sample['attention_mask'])
                }
                samples.append(sample)

            stats = self._analyze_masking_samples(samples)
            self._last_masking_stats = stats
            
            self.logger.info(
                f"Masking verification - Expected: {self.mlm_probability:.2%}, "
                f"Actual: {stats.average_ratio:.2%}"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in masking verification: {e}")
            raise

    def _analyze_masking_samples(self, samples: List[Dict]) -> MaskingStats:
        """Analyze masking statistics on multiple samples"""
        total_ratio = 0
        valid_samples = 0

        for sample in samples:
            masked_batch = self.data_collator([sample])
            attention_mask = sample['attention_mask']
            
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
                
            masked_tokens = (masked_batch['labels'][0] != -100).sum().item()
            valid_tokens = attention_mask.sum().item()
            
            if valid_tokens > 0:
                total_ratio += masked_tokens / valid_tokens
                valid_samples += 1

        average_ratio = total_ratio / valid_samples if valid_samples > 0 else 0
        return MaskingStats(
            average_ratio=average_ratio,
            num_samples=valid_samples,
            expected_ratio=self.mlm_probability,
            deviation=abs(average_ratio - self.mlm_probability)
        )

    def get_random_text(self, min_density: float = 0.6) -> str:
        """Get random text sample with minimum token density"""
        if not self.is_ready():
            raise ValueError("Dataset non chargé. Veuillez d'abord charger un dataset.")
        
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                # Prendre un échantillon aléatoire
                sample = next(iter(self.dataset.shuffle().take(1)))
                attention_mask = torch.tensor(sample["attention_mask"])
                density = attention_mask.sum().item() / len(attention_mask)
                
                if density >= min_density:
                    return self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                
                if attempt == max_attempts - 1:
                    self.logger.warning(
                        f"Impossible de trouver un texte avec une densité >= {min_density} "
                        f"après {max_attempts} tentatives"
                    )
                    
            except Exception as e:
                self.logger.error(f"Erreur lors de la sélection du texte: {e}")
                raise ValueError("Erreur lors de l'accès au dataset")
                
        raise ValueError(
            f"Impossible de trouver un texte avec une densité >= {min_density}. "
            f"Essayez de réduire la densité minimale requise."
        )

    def is_ready(self) -> bool:
        """Verify if streaming dataset is ready"""
        if not self.dataset:
            self.logger.warning("Dataset not loaded")
            return False
            
        if not self._dataset_size > 0:
            self.logger.warning("Dataset size is 0")
            return False
            
        try:
            # Verify streaming functionality
            next(iter(self.dataset))
            return True
        except Exception as e:
            self.logger.warning(f"Dataset not properly initialized: {e}")
            return False

    @property
    def dataset_size(self) -> int:
        """Get current dataset size in tokens"""
        return self._dataset_size

    def _format_loading_status(self, size_gb: float, stats: MaskingStats, buffer_size: int) -> str:
        """Format loading status with streaming metrics"""
        return (f"✅ Dataset streaming initialized successfully!\n"
                f"Target size: {size_gb:.1f} GB\n"
                f"Shuffle buffer: {buffer_size:,} examples\n"
                f"Sample masking ratio: {stats.average_ratio:.2%} (target: {self.mlm_probability:.1%})")

    
def extract_text(example):
    """Extract text with proper type handling"""
    try:
        if 'text' in example:
            if isinstance(example['text'], list):
                # mOSCAR format with list
                texts = []
                for item in example['text']:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
                example['text'] = ' '.join(texts)
            elif isinstance(example['text'], dict):
                # mOSCAR format avec dict
                example['text'] = example['text'].get('text', '')
            elif isinstance(example['text'], str):
                # Simple string format
                example['text'] = example['text']
        
        # Handle unexpected types for 'id' and 'meta'
        if 'id' in example and not isinstance(example['id'], str):
            example['id'] = str(example['id'])
        if 'meta' in example and not isinstance(example['meta'], dict):
            example['meta'] = {}
        
        return example
    except Exception as e:
        logging.warning(f"Error processing example: {e}")
        return {}