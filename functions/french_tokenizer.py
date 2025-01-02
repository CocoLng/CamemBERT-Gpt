import logging
from typing import Optional
from pathlib import Path
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import numpy as np

class FrenchTokenizerTrainer:
    """Gestionnaire d'entraînement du tokenizer pour le français"""
    
    def __init__(self, vocab_size: int = 32000):
        self.logger = logging.getLogger(__name__)
        self.vocab_size = vocab_size
        
    def prepare_training_data(self, output_path: str):
        """Prépare les données pour l'entraînement du tokenizer"""
        try:
            # Charge le dataset OSCAR français en streaming
            dataset = load_dataset(
                "oscar",
                "unshuffled_deduplicated_fr",
                streaming=True,
                split="train"
            )
            
            # Échantillonne 10^7 phrases aléatoirement
            self.logger.info("Sampling 10M sentences for tokenizer training...")
            sampled_texts = []
            for i, example in enumerate(dataset):
                if i >= 10_000_000:  # 10^7 phrases
                    break
                if isinstance(example["text"], str):
                    sampled_texts.append(example["text"])
            
            # Sauvegarde les textes pour l'entraînement
            output_file = Path(output_path) / "tokenizer_training_data.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with output_file.open("w", encoding="utf-8") as f:
                for text in sampled_texts:
                    f.write(text + "\n")
                    
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise

    def train_tokenizer(self, input_file: str, output_path: str):
        try:
            model_prefix = str(Path(output_path) / "french_sp")
            
            # Configuration améliorée de SentencePiece pour CamemBERT
            spm.SentencePieceTrainer.train(
                input=input_file,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                character_coverage=0.99999,  # Couverture caractères très élevée pour le français
                model_type="unigram",        # Comme spécifié dans l'article
                add_dummy_prefix=False,      # Pas de préfixe fictif
                normalization_rule_name="identity",  # Pas de normalisation
                user_defined_symbols=["<s>","</s>","<pad>","<mask>","<unk>"],
                # Ajouts recommandés:
                input_sentence_size=10000000,  # Pour les 10^7 phrases
                shuffle_input_sentence=True,   # Mélange des phrases
                train_extremely_large_corpus=True,  # Pour OSCAR
                num_threads=16  # Parallélisation
            )
            
            return model_prefix + ".model", model_prefix + ".vocab"
            
        except Exception as e:
            self.logger.error(f"Error training tokenizer: {e}")
            raise
            
    def load_tokenizer(self, model_path: str) -> PreTrainedTokenizerFast:
        """Charge le tokenizer entraîné en tant que PreTrainedTokenizerFast"""
        try:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=model_path,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                padding_side="right",
                truncation_side="right",
                model_max_length=512
            )
            
            # Configure les tokens spéciaux
            special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>"
            }
            tokenizer.add_special_tokens(special_tokens)
            
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise