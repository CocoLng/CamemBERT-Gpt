import logging
import os
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

class FineTuningConfig:
    """Configuration for fine-tuning tasks"""
    def __init__(
        self,
        model_path: str,
        dataset_name: str,
        num_labels: int,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3
    ):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

class FineTuning:
    """Manages fine-tuning of pre-trained models"""
    
    AVAILABLE_DATASETS = {
        "multi_nli": {
            "name": "nyu-mll/multi_nli",
            "num_labels": 3,
            "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"}
        },
        "xnli": {
            "name": "xnli",
            "num_labels": 3,
            "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"}
        }
    }
    
    def __init__(self):
        """Initialize fine-tuning manager"""
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.config = None
        self.current_dataset = None
        self.label_map = None

    def _is_valid_model_directory(self, directory: str) -> bool:
        """Vérifie si un dossier contient un modèle valide"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        return all(os.path.exists(os.path.join(directory, f)) for f in required_files)

    def load_model_for_fine_tuning(
        self,
        source: str,
        runs_dir: str = None,
        checkpoint: str = None,
        hf_model: str = None
    ) -> str:
        """Load model for fine-tuning from local checkpoint or HuggingFace"""
        try:
            if source == "Checkpoint Local":
                if not runs_dir or not checkpoint:
                    return "❌ Veuillez sélectionner un run et un checkpoint"
                    
                # Construire le chemin complet
                if checkpoint == "weights":
                    path = os.path.join(runs_dir, "weights")
                else:
                    path = os.path.join(runs_dir, checkpoint)
                
                if not os.path.exists(path):
                    return f"❌ Chemin non trouvé: {path}"
                    
                if not self._is_valid_model_directory(path):
                    return f"❌ Dossier de modèle invalide: {path}"
                    
            else:  # Modèle HuggingFace
                if not hf_model:
                    return "❌ Veuillez spécifier un nom de modèle HuggingFace"
                path = hf_model
            
            # Charger le modèle
            return self.load_model(path)
            
        except Exception as e:
            error_msg = f"❌ Erreur lors du chargement: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def load_model(self, model_path: str) -> str:
        """Load pre-trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3  # Default for NLI tasks
            )
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            return "✅ Modèle chargé avec succès"
            
        except Exception as e:
            error_msg = f"❌ Erreur lors du chargement du modèle: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def prepare_dataset(self, dataset_name: str) -> str:
        """Load and prepare dataset for fine-tuning"""
        try:
            if dataset_name not in self.AVAILABLE_DATASETS:
                return f"❌ Dataset {dataset_name} non supporté"

            dataset_info = self.AVAILABLE_DATASETS[dataset_name]
            self.label_map = dataset_info["label_map"]
            
            # Load dataset
            self.current_dataset = load_dataset(dataset_info["name"])
            
            # Tokenize dataset
            def preprocess_function(examples):
                return self.tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    max_length=128,
                    padding="max_length"
                )

            tokenized_dataset = self.current_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=self.current_dataset["train"].column_names
            )

            self.current_dataset = tokenized_dataset
            return "✅ Dataset préparé avec succès"
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de la préparation du dataset: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def start_fine_tuning(
        self,
        output_dir: str,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        wandb_project: str
    ) -> str:
        """Start fine-tuning process"""
        try:
            if not all([self.model, self.tokenizer, self.current_dataset]):
                return "❌ Veuillez d'abord charger un modèle et préparer le dataset"

            # Initialize wandb
            wandb.init(project=wandb_project, name=f"fine-tuning-{output_dir}")

            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                report_to="wandb",
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=100,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=4 if torch.cuda.is_available() else 0,
                dataloader_pin_memory=torch.cuda.is_available()
            )

            # Compute metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = predictions.argmax(-1)
                accuracy = (predictions == labels).mean()
                return {"accuracy": accuracy}

            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.current_dataset["train"],
                eval_dataset=self.current_dataset["validation"],
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(self.tokenizer),
                compute_metrics=compute_metrics
            )

            # Start training
            self.trainer.train()
            self.trainer.save_model()
            
            # Clean up wandb
            wandb.finish()
            
            return "✅ Fine-tuning terminé avec succès"
            
        except Exception as e:
            error_msg = f"❌ Erreur lors du fine-tuning: {str(e)}"
            self.logger.error(error_msg)
            if wandb.run is not None:
                wandb.finish()
            return error_msg

    def evaluate_model(self) -> Tuple[str, np.ndarray]:
        """Evaluate model and generate confusion matrix"""
        try:
            if not self.trainer:
                return "❌ Pas de modèle entraîné à évaluer", None

            # Get predictions
            predictions = self.trainer.predict(self.current_dataset["test"])
            preds = predictions.predictions.argmax(-1)
            labels = predictions.label_ids

            # Compute confusion matrix
            cm = confusion_matrix(labels, preds)
            
            # Create confusion matrix visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[self.label_map[i] for i in range(len(self.label_map))],
                yticklabels=[self.label_map[i] for i in range(len(self.label_map))]
            )
            plt.title('Matrice de Confusion')
            plt.ylabel('Vrai label')
            plt.xlabel('Prédiction')
            
            # Save plot
            plt.savefig('confusion_matrix.png')
            plt.close()

            # Get detailed classification report
            report = classification_report(
                labels,
                preds,
                target_names=[self.label_map[i] for i in range(len(self.label_map))],
                digits=3
            )

            accuracy = (preds == labels).mean()
            return (
                f"✅ Évaluation terminée\n"
                f"Précision globale: {accuracy:.2%}\n\n"
                f"Rapport détaillé:\n{report}",
                'confusion_matrix.png'
            )
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'évaluation: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None