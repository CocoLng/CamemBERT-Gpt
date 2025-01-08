import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .data_loader import DatasetConfig


@dataclass
class MaskingStats:
    """Structure pour les statistiques de masquage"""

    average_ratio: float
    num_samples: int
    expected_ratio: float
    deviation: float


class MaskingMonitorCallback(TrainerCallback):
    """Callback simplifié pour surveiller le masquage pendant l'entraînement"""

    def __init__(self, tokenizer, expected_mlm_probability: float = 0.15):
        self.logger = logging.getLogger(__name__)
        self.mask_token_id = tokenizer.mask_token_id
        self.expected_ratio = expected_mlm_probability
        self.check_frequency = 2000

        # Statistiques essentielles
        self.total_masks = 0
        self.total_tokens = 0
        self.current_batch_ratio = 0.0

    def analyze_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyse le ratio de masquage dans le lot courant"""
        try:
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))

            valid_tokens = attention_mask.sum().item()
            masked_tokens = (input_ids == self.mask_token_id).sum().item()

            self.total_masks += masked_tokens
            self.total_tokens += valid_tokens
            self.current_batch_ratio = (
                masked_tokens / valid_tokens if valid_tokens > 0 else 0
            )

            return {
                "current_masking_ratio": self.current_batch_ratio,  
                "expected_ratio": self.expected_ratio,
                "total_tokens": self.total_tokens,
                "total_masks": self.total_masks,
            }

        except Exception as e:
            self.logger.error(f"Erreur d'analyse du lot: {e}")
            return None

    def get_stats_dict(self) -> Dict[str, float]:
        """Renvoie les statistiques actuelles de masquage"""
        return {
            "current_masking_ratio": self.current_batch_ratio,
            "expected_ratio": self.expected_ratio,
            "total_tokens": self.total_tokens,
            "total_masks": self.total_masks,
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialise la surveillance"""
        self.logger.info(
            f"Début de la surveillance du masquage:\n"
            f"- Ratio attendu: {self.expected_ratio:.2%}\n"
            f"- Fréquence de vérification: {self.check_frequency} étapes"
        )

        if wandb.run is not None:
            wandb.define_metric("masking/current_masking_ratio")
            wandb.define_metric("masking/total_masks")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Surveille les étapes d'entraînement"""
        if state.global_step % self.check_frequency == 0:
            # Récupérer les inputs depuis le trainer
            trainer = kwargs.get("trainer")
            if trainer and hasattr(trainer, "_current_inputs"):
                inputs = trainer._current_inputs
                stats = self.analyze_batch(inputs)

                if stats and wandb.run is not None:
                    wandb.log(
                        {
                            "masking/current_masking_ratio": stats[
                                "current_masking_ratio"
                            ],
                            "masking/total_masks": stats["total_masks"],
                        },
                        step=state.global_step,
                    )


class MaskingHandler:
    def __init__(self, data_loader):
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        self.mlm_probability = 0.15  
        self.data_collator = self._initialize_data_collator()
        self.current_masking_ratio = 0.0  # Ajout de l'attribut

    def _initialize_data_collator(self) -> DataCollatorForLanguageModeling:
        """Initialise le collateur pour le masquage"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.data_loader.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=8,
        )

    def initialize_and_load_dataset(self, choice: str, size: float, prob: float) -> str:
        """Initialise et charge le dataset avec les paramètres spécifiés"""
        try:
            dataset_mapping = {
                "mOSCAR (default)": "oscar-corpus/mOSCAR",
                "OSCAR-2301": "oscar-corpus/OSCAR-2301",
            }
            dataset_name = dataset_mapping.get(choice)
            if not dataset_name:
                return f"❌ Dataset '{choice}' non reconnu."

            subset = "fra_Latn" if choice == "mOSCAR (default)" else "fr"

            # Met à jour la configuration du dataset
            self.data_loader.dataset_config = DatasetConfig(
                name=dataset_name, subset=subset, split="train", streaming=True
            )
            self.mlm_probability = prob
            self.data_collator = self._initialize_data_collator()

            status = self.load_streaming_dataset(size)
            logging.info("Dataset chargé avec succès")
            return status

        except Exception as e:
            return f"❌ Erreur: {str(e)}"

    def load_streaming_dataset(self, size_gb: float) -> str:
        """Charge le dataset en streaming avec le masquage"""
        try:
            # Calcul de la taille du dataset en tokens
            bytes_per_token = 4
            tokens_per_gb = int((1024 * 1024 * 1024) / bytes_per_token)
            self.data_loader._dataset_size = int(size_gb * tokens_per_gb)

            # Calcul du buffer optimal
            optimal_buffer = DatasetConfig.calculate_optimal_buffer(size_gb)

            # Chargement du dataset en streaming
            self.data_loader.dataset = load_dataset(
                self.data_loader.dataset_config.name,
                name=self.data_loader.dataset_config.subset,
                split=self.data_loader.dataset_config.split,
                streaming=True,
                trust_remote_code=True,
            )

            if not self.data_loader.dataset:
                raise ValueError(
                    f"Échec du chargement du dataset {self.data_loader.dataset_config.name}"
                )

            # Configuration du pipeline de streaming avec extraction des données
            self.data_loader.dataset = self.data_loader.dataset.shuffle(
                buffer_size=optimal_buffer
            )
            self.data_loader.dataset = self.data_loader.dataset.map(
                extract_text,
                remove_columns=[
                    col
                    for col in self.data_loader.dataset.column_names
                    if col != "text"
                ],
            )

            # Vérification de la présence du texte
            sample = next(iter(self.data_loader.dataset))
            if "text" not in sample or not sample["text"]:
                raise ValueError("Échec de l'extraction du texte du dataset")

            # Application de la tokenisation
            self.data_loader.dataset = self.data_loader.dataset.map(
                self.data_loader._tokenize_function,
                batched=True,
                remove_columns=["text"],
            )

            # Vérification initiale des échantillons
            masking_stats = self._verify_streaming_masking()
            self.logger.info(
                f"Dataset {self.data_loader.dataset_config.name} chargé avec succès avec une taille de {size_gb} GB"
            )

            return self._format_loading_status(size_gb, masking_stats, optimal_buffer)

        except Exception as e:
            error_msg = f"Échec du chargement du dataset: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return f"❌ {error_msg}"

    def _verify_streaming_masking(self) -> MaskingStats:
        """Vérifie le masquage sur les données en streaming"""
        try:
            total_ratio = 0
            samples_checked = 0
            max_samples = 5

            iterator = iter(self.data_loader.dataset)
            for _ in range(max_samples):
                try:
                    sample = next(iterator)
                    required_keys = {
                        "input_ids",
                        "attention_mask",
                        "special_tokens_mask",
                    }
                    if not all(k in sample for k in required_keys):
                        continue

                    sample_tensor = {
                        k: torch.tensor(v, dtype=torch.long)
                        for k, v in sample.items()
                        if isinstance(v, (list, np.ndarray))
                    }

                    if not all(k in sample_tensor for k in required_keys):
                        continue

                    masked_batch = self.data_collator([sample_tensor])

                    attention_mask = sample_tensor["attention_mask"]
                    masked_tokens = (masked_batch["labels"][0] != -100).sum().item()
                    valid_tokens = attention_mask.sum().item()

                    if valid_tokens > 0:
                        total_ratio += masked_tokens / valid_tokens
                        samples_checked += 1

                except StopIteration:
                    break
                except Exception as e:
                    self.logger.warning(
                        f"Erreur lors du traitement de l'échantillon: {e}"
                    )
                    continue

            if samples_checked == 0:
                raise ValueError(
                    "Aucun échantillon valide trouvé pour la vérification du masquage"
                )

            avg_ratio = total_ratio / samples_checked

            return MaskingStats(
                average_ratio=avg_ratio,
                num_samples=samples_checked,
                expected_ratio=self.mlm_probability,
                deviation=abs(avg_ratio - self.mlm_probability),
            )

        except Exception as e:
            self.logger.error(
                f"Erreur dans la vérification du masquage: {e}", exc_info=True
            )
            raise

    def visualize_with_density(self, text: str, density: float) -> Tuple[str, str]:
        """Visualise le masquage avec une densité minimale"""
        try:
            if text.strip():
                return self.visualize_masking(text)
            else:
                sample = next(iter(self.data_loader.dataset))
                text = self.data_loader.tokenizer.decode(
                    sample["input_ids"], skip_special_tokens=True
                )

            tokens = self.data_loader.tokenizer(text, truncation=True, max_length=512)
            current_density = sum(tokens["attention_mask"]) / len(
                tokens["attention_mask"]
            )

            if current_density >= density:
                return self.visualize_masking(text)
            else:
                return "Densité de texte trop faible", ""

        except Exception as e:
            self.logger.error(f"Erreur dans visualize_with_density: {e}")
            return f"Erreur: {str(e)}", ""

    def visualize_masking(self, text: str) -> Tuple[str, str]:
        """Visualise le masquage du texte d'entrée"""
        try:
            if not text.strip():
                return "❌ Texte vide", ""

            inputs = self.data_loader.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )

            masked = self.data_collator(
                [
                    {
                        "input_ids": inputs["input_ids"][0],
                        "attention_mask": inputs["attention_mask"][0],
                    }
                ]
            )

            original_text = self.data_loader.tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=True
            )

            masked_text = self.data_loader.tokenizer.decode(
                masked["input_ids"][0], skip_special_tokens=False
            )

            return original_text, masked_text

        except Exception as e:
            error_msg = f"❌ Erreur lors du masquage: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, ""

    def _format_loading_status(
        self, size_gb: float, stats: MaskingStats, buffer_size: int
    ) -> str:
        """Formate le statut du chargement du dataset"""
        return (
            f"✅ Dataset streaming initialisé avec succès !\n"
            f"Taille cible : {size_gb:.1f} GB\n"
            f"Taille du buffer de mélange : {buffer_size:,} exemples\n"
            f"Ratio de masquage des échantillons : {stats.average_ratio:.2%} (cible : {self.mlm_probability:.1%})"
        )

    def data_collator(self, examples):
        batch = self.data_loader.data_collator(examples)
        if 'labels' in batch:
            # Calcul du ratio de masquage actuel
            mask_tokens = (batch['labels'] != -100).sum().item()
            total_tokens = batch['labels'].numel()
            self.current_masking_ratio = mask_tokens / total_tokens if total_tokens > 0 else 0.0
        return batch


def extract_text(example):
    """Extrait le texte avec une gestion appropriée des types"""
    try:
        if "text" in example:
            if isinstance(example["text"], list):
                texts = []
                for item in example["text"]:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                return {"text": " ".join(texts)}
            elif isinstance(example["text"], dict):
                return {"text": example["text"].get("text", "")}
            elif isinstance(example["text"], str):
                return {"text": example["text"]}
        return {"text": ""}
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du texte: {e}")
        return {"text": ""}
