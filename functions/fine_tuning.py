import logging
import os

from typing import Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    CamembertTokenizer,
    Trainer,
    TrainingArguments,
)

from .training_saver import TrainingSaver

logger = logging.getLogger(__name__)


class CustomNLIModel(nn.Module):
    """Custom model with a classification head for NLI tasks."""

    def __init__(self, config, weights, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=weights,
        )

        hidden_size = config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(p=0.2)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_size // 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs[1]
        x = self.dropout(hidden_states)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits if loss is None else (loss, logits)


class CustomNLITrainer(TrainingSaver, Trainer):
    """Custom trainer that integrates TrainingSaver functionality."""

    def __init__(self, checkpoint_steps=1000, **kwargs):
        self.checkpoint_steps = checkpoint_steps
        self.tokens_processed = 0

        # Initialize TrainingSaver with NLI-specific paths
        TrainingSaver.__init__(
            self,
            run_dir=kwargs["args"].output_dir,
            dataset_size=kwargs.get("dataset_size", 0),
            processing_class=kwargs.get("tokenizer", None),
        )

        # Initialize the Trainer
        Trainer.__init__(self, **kwargs)

        # Save initial model info
        if hasattr(self, "model") and self.model is not None:
            self._save_model_info(self.run_dir)


class Finetune_NLI:
    """Fine-tune an NLI model for Natural Language Inference tasks."""
    def __init__(self, model_repo, weights_filename, config_filename, voca_filename, tokenizer_name="camembert-base"):
        self.logger = logging.getLogger(__name__)
        self.model_repo = model_repo
        self.voca_filename = voca_filename
        self.weights_filename = weights_filename
        self.config_filename = config_filename
        self.tokenizer_name = tokenizer_name
        self.tokenizer = CamembertTokenizer.from_pretrained(self.tokenizer_name)
        self.model = None
        self.config = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.trainer = None

    def _setup_run_dir(self) -> str:
        """Create and configure the run directory."""
        os.makedirs(self.base_dir, exist_ok=True)
        run_id = 0
        while True:
            run_dir = os.path.join(self.base_dir, f"nli_run{run_id}")
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
                return run_dir
            run_id += 1

    def load_data(self, dataset_name="facebook/xnli", language="fr", test_split=0.2):
        """Load and tokenize the dataset for NLI tasks."""
        logger.info("Loading dataset...")
        dataset = load_dataset(dataset_name, language, split="train")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        logger.info("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        train_dataset, test_dataset = tokenized_datasets.train_test_split(
            test_size=test_split, shuffle=True
        ).values()
        test_dataset, validation_dataset = test_dataset.train_test_split(
            test_size=0.5
        ).values()

        return train_dataset, test_dataset, validation_dataset

    def finetune_model(
        self,
        train_dataset,
        test_dataset,
        validation_dataset,
        num_epochs=1,
        batch=8,
        learning_rate=1e-5,
        num_labels=3,
    ):
        """Fine-tune the NLI model using CustomNLITrainer."""
        logger.info("Loading model configuration and weights...")

        # Download weights and config
        weights_path = hf_hub_download(self.model_repo, self.weights_filename)
        config_path = hf_hub_download(self.model_repo, self.config_filename)

        # Load config and weights
        self.config = AutoConfig.from_pretrained(config_path)
        weights = load_file(weights_path)

        # Initialize the custom model
        self.model = CustomNLIModel(self.config, weights, num_labels)

        # Define training arguments with proper directories
        training_args = TrainingArguments(
            output_dir=self.run_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=8,
            learning_rate=learning_rate,
            logging_dir=os.path.join(self.run_dir, "logs"),
            logging_steps=1,
            report_to="tensorboard",
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        # Initialize CustomNLITrainer
        logger.info("Initializing Trainer...")
        trainer = CustomNLITrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            dataset_size=len(train_dataset),
        )

        # Fine-tune the model
        logger.info("Starting training...")
        trainer.train()

        # Evaluate and save results
        logger.info("Evaluating model...")
        results = trainer.evaluate(validation_dataset)

        # Save the final model and results
        save_dir = os.path.join(self.run_dir, "final_model")
        os.makedirs(save_dir, exist_ok=True)

        trainer.save_model()
        logger.info(f"Model saved to {save_dir}")

        return trainer, results
    
    def initialize_model(self, model_repo: str, weights_path: str, config_path: str, tokenizer_name: str) -> str:
        """Initialize the model with given parameters."""
        try:
            self.model_repo = model_repo
            self.weights_filename = weights_path
            self.config_filename = config_path
            self.tokenizer_name = tokenizer_name
            
            # Update tokenizer if needed
            if self.tokenizer_name != tokenizer_name:
                self.tokenizer = CamembertTokenizer.from_pretrained(tokenizer_name)
            
            return "✅ Configuration du modèle mise à jour avec succès"
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            return f"❌ Erreur lors de l'initialisation: {str(e)}"

    def prepare_dataset(self, dataset_name: str, language: str, test_split: float) -> Tuple[str, pd.DataFrame]:
        """Prepare the dataset and return status and preview."""
        try:
            self.train_dataset, self.test_dataset, self.val_dataset = \
                self.load_data(dataset_name, language, test_split)
            
            # Create preview DataFrame
            preview = {
                'Split': ['Train', 'Test', 'Validation'],
                'Size': [
                    len(self.train_dataset),
                    len(self.test_dataset),
                    len(self.val_dataset)
                ]
            }
            preview_df = pd.DataFrame(preview)
            
            return "✅ Données préparées avec succès", preview_df
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            return f"❌ Erreur lors de la préparation des données: {str(e)}", pd.DataFrame()

    def evaluate_model(self) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:
        """Evaluate the model and return metrics and visualizations."""
        try:
            if not self.trainer:
                return pd.DataFrame(), None, None

            # Get evaluation results
            eval_results = self.trainer.evaluate(self.val_dataset)
            
            # Get predictions for confusion matrix
            predictions = self.trainer.predict(self.val_dataset)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            true_labels = predictions.label_ids

            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            # Create confusion matrix plot
            cm_fig = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matrice de Confusion')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Create learning curve plot
            history = self.trainer.state.log_history
            loss_history = [(log['step'], log['loss']) for log in history if 'loss' in log]
            steps, losses = zip(*loss_history)
            
            acc_fig = plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.title('Courbe d\'Apprentissage')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Metric': list(eval_results.keys()),
                'Value': list(eval_results.values())
            })

            return metrics_df, cm_fig, acc_fig

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return pd.DataFrame(), None, None

    def start_training(self, num_epochs: int, batch_size: int, learning_rate: float, 
                      num_labels: int, save_dir: str, use_tensorboard: bool) -> str:
        """Start the fine-tuning process."""
        try:
            if not all([self.train_dataset, self.val_dataset]):
                return "❌ Données non préparées. Veuillez d'abord préparer les données."

            trainer, results = self.finetune_model(
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                validation_dataset=self.val_dataset,
                num_epochs=num_epochs,
                batch=batch_size,
                learning_rate=learning_rate,
                num_labels=num_labels
            )
            
            self.trainer = trainer
            return "✅ Fine-tuning terminé avec succès"
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return f"❌ Erreur lors de l'entraînement: {str(e)}"

    def stop_training(self) -> str:
        """Stop the training process."""
        try:
            if self.trainer and hasattr(self.trainer, 'model'):
                # Save current state
                self.trainer.save_model()
                return "✅ Entraînement arrêté et modèle sauvegardé"
            return "ℹ️ Aucun entraînement en cours"
        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
            return f"❌ Erreur lors de l'arrêt de l'entraînement: {str(e)}"
