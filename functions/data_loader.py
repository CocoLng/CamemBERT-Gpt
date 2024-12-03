import logging
from datasets import load_dataset
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
import torch
import random
from typing import Dict, List, Tuple, Union
import numpy as np
import os

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = None
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
    def load_streaming_dataset(self, size_gb: float) -> str:
        """Charge une portion du dataset OSCAR en streaming"""
        try:
            # Approximation du nombre de tokens par GB
            tokens_per_gb = int((1024 * 1024 * 1024) / 4)  # ~4 bytes par token
            total_tokens = int(size_gb * tokens_per_gb)
            
            self.dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "fr",
                split="train",
                streaming=True
            ).take(total_tokens)
            
            # Transform dataset to include tokenized inputs
            self.dataset = self.dataset.map(
                lambda examples: self.tokenize_function(examples['text']),
                remove_columns=['text', 'meta']
            )
            
            return f"✅ Dataset chargé avec succès! Taille approximative: {size_gb} GB"
        except Exception as e:
            error_msg = f"❌ Erreur lors du chargement du dataset: {e}"
            self.logger.error(error_msg)
            return error_msg

    def tokenize_function(self, text: Union[str, List[str]]) -> Dict:
        """Tokenize text with proper truncation"""
        return self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_special_tokens_mask=True
        )

    def get_random_text(self) -> str:
        """Récupère un texte aléatoire du dataset"""
        if not self.dataset:
            return "Veuillez d'abord charger le dataset."
        
        try:
            sample = next(iter(self.dataset.shuffle(seed=random.randint(0, 1000)).take(1)))
            return self.tokenizer.decode(sample['input_ids'])
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du texte: {e}")
            return "Erreur lors de la récupération du texte."

    def _mask_single_text(self, text: str) -> Dict:
        """Handle single text input for visualization"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            
            # Format for data collator
            features = [{
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
            }]
            
            # Apply masking using the data collator
            masked = self.data_collator(features)
            
            return masked
            
        except Exception as e:
            self.logger.error(f"Error in masking text: {e}")
            raise

    def visualize_masking(self, text: str) -> Tuple[str, str]:
        """Visualise le masquage appliqué sur un texte"""
        try:
            if not text.strip():
                text = self.get_random_text()
                
            # Apply masking
            masked_inputs = self._mask_single_text(text)
            masked_text = self.tokenizer.decode(masked_inputs["input_ids"][0])
            
            return text, masked_text
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {e}")
            return text, f"Error in masking: {str(e)}"