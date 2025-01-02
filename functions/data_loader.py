import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import psutil
from datasets import Dataset, load_dataset
from transformers import RobertaTokenizerFast

from .french_tokenizer import FrenchTokenizerTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DatasetConfig:
    """Configuration pour le chargement du dataset avec gestion optimisée du buffer"""

    name: str = "oscar"
    subset: Optional[str] = "unshuffled_deduplicated_fr"
    split: str = "train"
    streaming: bool = True
    verification_mode: str = "no_checks"

    @staticmethod
    def calculate_optimal_buffer(target_size_gb: float) -> int:
        """Calcule la taille optimale du buffer basée sur la mémoire disponible et la taille du dataset"""
        try:
            mem_available = psutil.virtual_memory().available / (1024**3)

            if (target_size_gb < 5):
                base_buffer = 25000
            elif (target_size_gb < 20):
                base_buffer = 50000
            else:
                base_buffer = 100000

            mem_factor = min(mem_available * 0.15 / target_size_gb, 1.0)
            return int(base_buffer * mem_factor)
        except Exception as e:
            logging.warning(f"Buffer size calculation failed: {e}, using default")
            return 10000


class DataLoader:
    """Gère le chargement et le prétraitement du dataset pour l'entraînement de CamemBERT"""

    def __init__(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        tokenizer_path: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.dataset_config = dataset_config or DatasetConfig()
        
        if tokenizer_path:
            # Utilise un tokenizer existant
            self.tokenizer = FrenchTokenizerTrainer().load_tokenizer(tokenizer_path)
        else:
            # Entraîne un nouveau tokenizer
            trainer = FrenchTokenizerTrainer()
            data_file = trainer.prepare_training_data("tokenizer_data")
            model_path, _ = trainer.train_tokenizer(data_file, "tokenizer")
            self.tokenizer = trainer.load_tokenizer(model_path)
            
        self.dataset = None
        self._dataset_size = 0

    def _initialize_tokenizer(self) -> RobertaTokenizerFast:
        """Initialise le tokenizer RoBERTa"""
        try:
            return RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def _tokenize_function(self, examples: Dict) -> Dict:
        """Tokenisation spécifique à CamemBERT avec masquage par mots entiers"""
        try:
            if not isinstance(examples["text"], (str, list)):
                self.logger.warning(f"Unexpected text type: {type(examples['text'])}")
                if isinstance(examples["text"], dict) and "text" in examples["text"]:
                    examples["text"] = examples["text"]["text"]

            # Ajout du masquage par mots entiers (WWM)
            # On utilise les espaces comme délimiteurs de mots
            words = examples["text"].split()
            mask_indices = np.random.rand(len(words)) < 0.15
            
            for idx in range(len(words)):
                if mask_indices[idx]:
                    r = np.random.rand()
                    if r < 0.8:
                        words[idx] = "<mask>"
                    elif r < 0.9:
                        words[idx] = words[idx]  # Keep unchanged 10% of time
                    else:
                        # Random word 10% of time
                        words[idx] = np.random.choice(words)
                        
            masked_text = " ".join(words)

            tokenized = self.tokenizer(
                masked_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_special_tokens_mask=True,
                return_attention_mask=True,
            )

            return tokenized

        except Exception as e:
            self.logger.error(f"Tokenization error: {e}", exc_info=True)
            raise

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prépare le dataset avec l'extraction du texte et la tokenisation appropriées"""
        try:
            # Extraire le texte et tokenizer
            dataset = dataset.map(self._extract_text)
            dataset = dataset.select_columns(["text"])
            dataset = dataset.map(
                self._tokenize_function, batched=True, remove_columns=["text"]
            )
            return dataset

        except Exception as e:
            self.logger.error(f"Dataset preparation error: {e}")
            raise

    @staticmethod
    def _extract_text(example: Dict) -> Dict:
        """Extrait le contenu textuel de différents formats de dataset"""
        try:
            if "text" in example:
                if isinstance(example["text"], list):
                    texts = [
                        item["text"] if isinstance(item, dict) else item
                        for item in example["text"]
                    ]
                    return {"text": " ".join(texts)}
                elif isinstance(example["text"], dict):
                    return {"text": example["text"].get("text", "")}
                elif isinstance(example["text"], str):
                    return {"text": example["text"]}
            return {"text": ""}
        except Exception as e:
            logging.error(f"Text extraction error: {e}")
            return {"text": ""}

    def load_dataset(self, size_gb: float) -> bool:
        """Charge le dataset avec une taille spécifiée en GB"""
        try:
            # Calcule le nombre total de tokens à charger
            bytes_per_token = 4
            tokens_per_gb = int((1024 * 1024 * 1024) / bytes_per_token)
            total_tokens = int(size_gb * tokens_per_gb)
            self._dataset_size = total_tokens

            # Calcule le buffer optimal
            optimal_buffer = DatasetConfig.calculate_optimal_buffer(size_gb)

            # Charge le dataset en streaming
            raw_dataset = load_dataset(
                self.dataset_config.name,
                name=self.dataset_config.subset,
                split=self.dataset_config.split,
                streaming=True,
                trust_remote_code=True,
            )

            if not raw_dataset:
                raise ValueError(f"Échec du chargement du dataset {self.dataset_config.name}")

            # Estimation du nombre moyen de tokens par exemple après tokenisation
            average_tokens_per_example = 256  # Vous pouvez ajuster cette valeur si nécessaire

            # Calcule le nombre d'exemples à prendre pour atteindre le total_tokens
            num_examples = total_tokens // average_tokens_per_example

            # Applique un shuffle et limite le dataset au nombre d'exemples calculé
            raw_dataset = raw_dataset.shuffle(buffer_size=optimal_buffer)
            limited_dataset = raw_dataset.take(num_examples)

            # Prépare le dataset limité
            self.dataset = self.prepare_dataset(limited_dataset)

            return True

        except Exception as e:
            self.logger.error(f"Erreur de chargement du dataset: {e}")
            return False

    def is_ready(self) -> bool:
        """Vérifie si le dataset est correctement chargé et prêt à l'utilisation"""
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
        """Obtient la taille actuelle du dataset en tokens"""
        return self._dataset_size
