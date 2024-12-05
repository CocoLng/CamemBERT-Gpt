import logging
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast

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
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = self._initialize_tokenizer()
        self.dataset = None
        self.mlm_probability = 0.15
        self._dataset_size = 0
        self.data_collator = self._initialize_data_collator()
        self._last_masking_stats = None

    def _initialize_tokenizer(self) -> RobertaTokenizerFast:
        """Initialize RoBERTa tokenizer with error handling"""
        try:
            return RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def _initialize_data_collator(self) -> DataCollatorForLanguageModeling:
        """Initialize MLM data collator with current probability"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability
        )

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

    def load_streaming_dataset(self, size_gb: float) -> str:
        """Load streaming dataset with verification"""
        try:
            tokens_per_gb = int((1024 * 1024 * 1024) / 4)
            self._dataset_size = int(size_gb * tokens_per_gb)
            
            self.dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "fr",
                split="train",
                streaming=True
            ).shuffle(buffer_size=10000)
            
            self.dataset = self.dataset.map(
                self._process_text,
                remove_columns=["text", "meta"],
                batched=True
            ).take(self._dataset_size)
            
            # Verify masking
            masking_stats = self._verify_masking()
            return self._format_loading_status(size_gb, masking_stats)
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    def _process_text(self, examples: Dict) -> Dict:
        """Process text examples with tokenization"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_special_tokens_mask=True
        )

    def visualize_with_density(self, text: str, density: float) -> Tuple[str, str]:
        """Visualize masking with minimum density requirement"""
        try:
            if text.strip():
                return self.visualize_masking(text)
            
            random_text = self.get_random_text(min_density=float(density))
            return self.visualize_masking(random_text)
            
        except Exception as e:
            error_msg = f"Error in visualization: {e}"
            self.logger.error(error_msg)
            return error_msg, ""

    def visualize_masking(self, text: str) -> Tuple[str, str]:
        """Visualize masking of input text"""
        try:
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
            self.logger.error(f"Error in masking visualization: {e}")
            return f"Error: {str(e)}", ""

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
            raise ValueError("Dataset not loaded")
        
        for _ in range(10):  # Max attempts
            sample = next(iter(self.dataset.shuffle().take(1)))
            attention_mask = torch.tensor(sample["attention_mask"])
            density = attention_mask.sum().item() / len(attention_mask)
            
            if density >= min_density:
                return self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                
        raise ValueError(f"Could not find text with density >= {min_density}")

    def is_ready(self) -> bool:
        """Check if dataset is loaded and ready"""
        return self.dataset is not None and self._dataset_size > 0

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