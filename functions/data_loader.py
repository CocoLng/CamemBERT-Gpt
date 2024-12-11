import logging
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast
from datasets import Dataset
from torch.utils.data import DataLoader
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    name: str = "oscar" 
    subset: Optional[str] = "unshuffled_deduplicated_fr"  # Default French subset
    split: str = "train"
    streaming: bool = True
    buffer_size: int = 10000
    trust_remote_code: bool = True
    verification_mode: str = "no_checks"

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
        self.mlm_probability = 0.15
        self._dataset_size = 0
        self.data_collator = self._initialize_data_collator()
        self.dataset_config = dataset_config or DatasetConfig()
        
        # Token processing configuration
        self.tokens_to_fuse = tokens_to_fuse or []
        self.tokens_to_remove = tokens_to_remove or []
        

    def _initialize_tokenizer(self) -> RobertaTokenizerFast:
        """Initialize RoBERTa tokenizer with error handling"""
        try:
            return RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def _initialize_data_collator(self) -> DataCollatorForLanguageModeling:
        """Initialize MLM data collator with proper configuration"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability
        )

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset with proper text extraction"""
        def extract_text(example):
            # Pour mOSCAR
            if isinstance(example.get('text', ''), list):
                texts = [item['text'] if isinstance(item, dict) else str(item) 
                        for item in example['text']]
                return {'text': ' '.join(texts)}
            # Pour OSCAR
            elif 'content' in example:
                return {'text': example['content']}
            # Cas où text est déjà au bon format
            elif isinstance(example.get('text', ''), str):
                return {'text': example['text']}
            return {'text': ''}

        dataset = dataset.map(extract_text)
        return dataset.select_columns(['text'])

    def _tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize text using batch processing"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_special_tokens_mask=True
        )

    def get_train_dataloader(self) -> DataLoader:
        """Create optimized streaming dataloader with proper batching"""
        try:
            if not self.dataset:
                raise ValueError("Dataset not initialized")
                
            return DataLoader(
                self.dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=2,
                drop_last=True  # Important pour éviter les problèmes de batch incomplet
            )
                
        except Exception as e:
            self.logger.error(f"Error creating dataloader: {e}")
            raise

    def load_streaming_dataset(self, size_gb: float) -> str:
        """Load and process streaming dataset"""
        try:
            tokens_per_gb = int((1024 * 1024 * 1024) / 4)
            self._dataset_size = int(size_gb * tokens_per_gb)

            # Load base dataset
            self.dataset = load_dataset(
                self.dataset_config.name,
                name=self.dataset_config.subset,
                split=self.dataset_config.split,
                streaming=True
            )

            if not self.dataset:
                raise ValueError(f"Failed to load dataset {self.dataset_config.name}")

            # Process dataset
            self.dataset = self.dataset.shuffle(buffer_size=10000)
            self.dataset = self._prepare_dataset(self.dataset)
            
            # Tokenize and limit size
            self.dataset = self.dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=['text']
            ).take(self._dataset_size)

            # Verify masking
            masking_stats = self._verify_masking()
            return self._format_loading_status(size_gb, masking_stats)

        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

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
        """Check if dataset is loaded and ready"""
        ready = (self.dataset is not None and 
                self._dataset_size > 0 and 
                next(iter(self.dataset.take(1)), None) is not None)
        
        if not ready:
            self.logger.warning("Dataset non chargé ou vide")
            
        return ready

    @property
    def dataset_size(self) -> int:
        """Get current dataset size in tokens"""
        return self._dataset_size

    def _format_loading_status(self, size_gb: float, stats: MaskingStats) -> str:
        """Format loading status message"""
        return (f"✅ Dataset loaded successfully! "
                f"Size: {size_gb} GB, "
                f"Effective masking: {stats.average_ratio:.2%} "
                f"(target: {self.mlm_probability:.1%})")