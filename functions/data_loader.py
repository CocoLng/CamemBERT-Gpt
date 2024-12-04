import logging
import os
import random
import torch
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast

@dataclass
class MaskingResult:
    """Structure pour les résultats de masquage"""
    original_text: str
    masked_text: str
    masking_ratio: float
    num_masked_tokens: int
    total_tokens: int

class DataLoaderError(Exception):
    """Classe personnalisée pour les erreurs du DataLoader"""
    pass

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_tokenizer()
        self.dataset = None
        self.mlm_probability = 0.15
        self._dataset_size = 0
        self._seed = 42
        self._last_masking_ratio = 0.0
        self._initialize_data_collator()

    def _setup_tokenizer(self) -> None:
        """Initialise le tokenizer avec gestion des erreurs"""
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception as e:
            raise DataLoaderError(f"Erreur d'initialisation du tokenizer: {e}")

    def _initialize_data_collator(self) -> None:
        """Initialise le data collator avec la probabilité MLM actuelle"""
        try:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability
            )
        except Exception as e:
            raise DataLoaderError(f"Erreur d'initialisation du data collator: {e}")

    def set_mlm_probability(self, prob: float) -> None:
        """Met à jour la probabilité de masquage"""
        if not 0 < prob < 1:
            raise ValueError("La probabilité de masquage doit être entre 0 et 1")
        if prob != self.mlm_probability:
            self.mlm_probability = prob
            self._initialize_data_collator()
            self.logger.info(f"Probabilité MLM mise à jour: {prob}")

    def load_streaming_dataset(self, size_gb: float, seed: Optional[int] = None) -> str:
        """Charge une portion du dataset OSCAR en streaming"""
        try:
            if seed is not None:
                self._seed = seed
            
            self._dataset_size = self._calculate_dataset_size(size_gb)
            self.dataset = self._setup_streaming_dataset()
            masking_stats = self._verify_masking()
            
            return self._format_loading_status(size_gb, masking_stats)
            
        except Exception as e:
            error_msg = f"Erreur lors du chargement du dataset: {e}"
            self.logger.error(error_msg)
            return error_msg

    def _calculate_dataset_size(self, size_gb: float) -> int:
        """Calcule la taille du dataset en tokens"""
        tokens_per_gb = int((1024 * 1024 * 1024) / 4)
        return int(size_gb * tokens_per_gb)

    def _setup_streaming_dataset(self):
        """Configure le dataset en streaming avec shuffling"""
        try:
            base_dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "fr",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            buffer_size = min(10000, self._dataset_size)
            shuffled_dataset = base_dataset.shuffle(
                seed=self._seed,
                buffer_size=buffer_size
            )

            processed_dataset = shuffled_dataset.map(
                lambda examples: self.tokenize_function(examples["text"]),
                remove_columns=["text", "meta"],
                batched=True,
                batch_size=1000
            )

            return processed_dataset.take(self._dataset_size)
        except Exception as e:
            raise DataLoaderError(f"Erreur lors de la configuration du dataset: {e}")

    def _verify_masking(self) -> Dict[str, float]:
        """Vérifie et analyse le masquage sur un échantillon"""
        try:
            # Prendre plusieurs échantillons pour l'analyse
            samples = []
            for sample in self.dataset.take(5):
                # Convertir les listes en tensors
                sample = {
                    'input_ids': torch.tensor(sample['input_ids']),
                    'attention_mask': torch.tensor(sample['attention_mask'])
                }
                samples.append(sample)

            stats = self._analyze_masking_samples(samples)
            self._last_masking_ratio = stats['average_ratio']
            
            self.logger.info(
                f"Vérification masquage - Attendu: {self.mlm_probability:.2%}, "
                f"Réel: {stats['average_ratio']:.2%}"
            )
            
            return stats
            
        except Exception as e:
            raise DataLoaderError(f"Erreur lors de la vérification du masking: {e}")

    def _analyze_masking_samples(self, samples: List[Dict]) -> Dict[str, float]:
        """Analyse les statistiques de masquage sur plusieurs échantillons"""
        total_ratio = 0
        valid_samples = 0

        for sample in samples:
            masked_batch = self.data_collator([sample])
            
            # Assurer que attention_mask est un tensor
            if not isinstance(sample['attention_mask'], torch.Tensor):
                attention_mask = torch.tensor(sample['attention_mask'])
            else:
                attention_mask = sample['attention_mask']
                
            stats = self._analyze_single_mask(masked_batch, attention_mask)
            
            if stats['valid_tokens'] > 0:
                total_ratio += stats['mask_ratio']
                valid_samples += 1

        average_ratio = total_ratio / valid_samples if valid_samples > 0 else 0
        return {
            'average_ratio': average_ratio,
            'num_samples': valid_samples
        }

    def _analyze_single_mask(self, masked_batch: Dict, attention_mask: torch.Tensor) -> Dict:
        """Analyse les statistiques de masquage pour un seul échantillon"""
        # S'assurer que les tenseurs sont sur le bon device
        attention_mask = attention_mask.to(masked_batch['labels'].device)
        
        # Calculer les statistiques
        valid_tokens = attention_mask.sum().item()
        masked_tokens = (masked_batch['labels'][0] != -100).sum().item()
        mask_ratio = masked_tokens / valid_tokens if valid_tokens > 0 else 0
        
        return {
            'valid_tokens': valid_tokens,
            'masked_tokens': masked_tokens,
            'mask_ratio': mask_ratio
        }

    def visualize_masking(self, text: str, min_density: float = 0.6) -> Tuple[str, str]:
        """
        Visualise le masquage d'un texte. Si aucun texte n'est fourni, en sélectionne un aléatoirement.
        
        Args:
            text (str): Texte à masquer (si vide, un texte aléatoire sera utilisé)
            min_density (float): Pour la sélection aléatoire, densité minimale de texte requise
            
        Returns:
            Tuple[str, str]: (texte original, texte masqué)
        """
        try:
            # Si pas de texte fourni, en prendre un aléatoire du dataset
            if not text.strip():
                if not self.dataset:
                    return "Veuillez d'abord charger le dataset.", ""
                    
                # Prendre des échantillons jusqu'à trouver un texte assez dense
                for _ in range(10):  # max 10 tentatives
                    sample = next(iter(self.dataset.shuffle(seed=random.randint(0, 1000)).take(1)))
                    density = sum(sample['attention_mask']) / len(sample['attention_mask'])
                    
                    if density >= min_density:
                        text = self.tokenizer.decode(sample["input_ids"])
                        break
                else:
                    return f"Impossible de trouver un texte avec densité >= {min_density}", ""

            # Appliquer le masquage
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

            masked_text = self.tokenizer.decode(masked["input_ids"][0])
            
            return text, masked_text

        except Exception as e:
            self.logger.error(f"Erreur lors de la visualisation du masking: {e}")
            return f"Erreur: {str(e)}", ""

    def get_random_text(self, min_density: float, max_attempts: int = 10) -> str:
        """Récupère un texte aléatoire avec une densité minimale"""
        if not self.is_ready():
            raise DataLoaderError("Dataset non chargé")
            
        for _ in range(max_attempts):
            sample = next(iter(self.dataset.shuffle(seed=random.randint(0, 1000)).take(1)))
            # Convertir en tensor pour le calcul de densité
            sample = {k: torch.tensor(v) for k, v in sample.items()}
            density = self._calculate_text_density(sample)
            if density >= min_density:
                return self.tokenizer.decode(sample["input_ids"])
        
        raise DataLoaderError(f"Impossible de trouver un texte avec densité >= {min_density}")

    def _apply_masking(self, text: str) -> Dict:
        """Applique le masquage à un texte"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors="pt"
        )

        features = [{
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
        }]

        return self.data_collator(features)

    def tokenize_function(self, texts: Union[str, List[str]]) -> Dict:
        """Tokenize le texte avec troncature et gestion des différents formats d'entrée"""
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise DataLoaderError("Format de texte non supporté")
            
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    def _calculate_text_density(self, sample: Dict[str, torch.Tensor]) -> float:
        """Calcule la densité de texte (ratio tokens réels vs padding)"""
        attention_mask = sample['attention_mask']
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        return attention_mask.sum().item() / len(attention_mask)

    def is_ready(self) -> bool:
        """Vérifie si le dataset est prêt"""
        return self.dataset is not None and self._dataset_size > 0

    def get_dataset_size(self) -> int:
        """Retourne la taille du dataset"""
        return self._dataset_size

    def _format_loading_status(self, size_gb: float, masking_stats: Dict[str, float]) -> str:
        """Formate le message de statut du chargement"""
        return (f"✅ Dataset chargé avec succès! "
                f"Taille: {size_gb} GB, "
                f"Masking effectif: {masking_stats['average_ratio']:.2%} "
                f"(cible: {self.mlm_probability:.1%})")