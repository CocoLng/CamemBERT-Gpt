import logging
from typing import Dict
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb

class MaskingMonitorCallback(TrainerCallback):
    """Simplified callback for monitoring masking during training"""
    
    def __init__(self, tokenizer, expected_mlm_probability: float = 0.15):
        self.logger = logging.getLogger(__name__)
        self.mask_token_id = tokenizer.mask_token_id
        self.expected_ratio = expected_mlm_probability
        self.check_frequency = 500  # Reduced frequency for better performance
        
        # Essential statistics
        self.total_masks = 0
        self.total_tokens = 0
        self.current_batch_ratio = 0.0
        
    def analyze_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze masking ratio in current batch"""
        try:
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            
            valid_tokens = attention_mask.sum().item()
            masked_tokens = (input_ids == self.mask_token_id).sum().item()
            
            self.total_masks += masked_tokens
            self.total_tokens += valid_tokens
            self.current_batch_ratio = masked_tokens / valid_tokens if valid_tokens > 0 else 0
            
            return {
                'current_masking_ratio': self.current_batch_ratio,  # Fixed key name
                'expected_ratio': self.expected_ratio,
                'total_tokens': self.total_tokens,
                'total_masks': self.total_masks
            }
            
        except Exception as e:
            self.logger.error(f"Batch analysis error: {e}")
            return None

    def get_stats_dict(self) -> Dict[str, float]:
        """Return current masking statistics"""
        return {
            'current_masking_ratio': self.current_batch_ratio,
            'expected_ratio': self.expected_ratio,
            'total_tokens': self.total_tokens,
            'total_masks': self.total_masks
        }

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Initialize monitoring"""
        self.logger.info(
            f"Starting masking monitoring:\n"
            f"- Expected ratio: {self.expected_ratio:.2%}\n"
            f"- Check frequency: {self.check_frequency} steps"
        )
        
        if wandb.run is not None:
            wandb.define_metric("masking/current_masking_ratio")
            wandb.define_metric("masking/total_masks")

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, **kwargs):
        """Monitor training steps"""
        if state.global_step % self.check_frequency == 0:
            stats = self.analyze_batch(kwargs.get('inputs', {}))
            
            if stats and wandb.run is not None:
                wandb.log({
                    "masking/current_masking_ratio": stats["current_masking_ratio"],
                    "masking/total_masks": stats["total_masks"]
                }, step=state.global_step)