import logging
import os
from typing import Tuple, Dict, Union, Any
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    CamembertTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForMaskedLM,
)
from datasets import load_dataset
from .fine_tuning_saver import FineTuningSaver

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


class CustomNLITrainer(Trainer):
    def __init__(self, checkpoint_steps=1000, saver=None, **kwargs):
        # Remove tokenizer from kwargs if present to avoid deprecation warning
        if 'tokenizer' in kwargs:
            del kwargs['tokenizer']
        super().__init__(**kwargs)
        self.checkpoint_steps = checkpoint_steps
        self.tokens_processed = 0
        self.saver = saver
        
        # Initialize saver if not provided
        if self.saver is None:
            try:
                output_dir = kwargs.get("args", TrainingArguments("default")).output_dir
                self.saver = FineTuningSaver(run_dir=output_dir)
            except Exception as e:
                logger.error(f"Failed to initialize saver: {e}")
                self.saver = None
        
    def training_step(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]], 
        return_loss=True
    ) -> torch.Tensor:
        """Override training step with checkpoint handling."""
        # Call parent's training step with all arguments
        loss = super().training_step(model, inputs, return_loss)
        
        if self.state.global_step > 0 and self.state.global_step % self.checkpoint_steps == 0:
            if self.saver:
                try:
                    metrics = self.evaluate()
                    self.saver.save_nli_checkpoint(self.state.global_step, metrics)
                    # Cleanup old checkpoints
                    self.saver.cleanup_old_checkpoints(max_checkpoints=5)
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {self.state.global_step}: {e}")
                    self._emergency_save(model)
        
        return loss
    
    def _emergency_save(self, model):
        """Emergency save in case of checkpoint failure."""
        try:
            emergency_dir = os.path.join(self.args.output_dir, "emergency_backup")
            os.makedirs(emergency_dir, exist_ok=True)
            model.save_pretrained(emergency_dir)
            logger.info(f"Emergency backup saved to {emergency_dir}")
        except Exception as e:
            logger.critical(f"Emergency save failed: {e}")


    def save_model(self, output_dir=None):
        """Override save_model to add extra safety checks."""
        try:
            # Save with parent method
            super().save_model(output_dir)
            
            # Verify saved files
            if output_dir is None:
                output_dir = self.args.output_dir
                
            required_files = ['config.json', 'pytorch_model.bin']
            missing_files = [f for f in required_files 
                           if not os.path.exists(os.path.join(output_dir, f))]
            
            if missing_files:
                raise ValueError(f"Missing required files after save: {missing_files}")
                
            # Save evaluation metrics
            if hasattr(self, 'eval_metrics'):
                self.saver.save_evaluation_metrics(self.eval_metrics)
                
        except Exception as e:
            self.logger.error(f"Error during model saving: {e}")
            # Try to save to backup location
            backup_dir = os.path.join(output_dir, 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            super().save_model(backup_dir)
            raise


class Finetune_NLI:
    """Fine-tune an NLI model for Natural Language Inference tasks."""
    def __init__(self, model_repo=None, weights_filename=None, config_filename=None, voca_filename=None, base_dir="camembert-nli"):
        self.logger = logging.getLogger(__name__)
        self.model_repo = model_repo
        self.voca_filename = voca_filename
        self.weights_filename = weights_filename
        self.config_filename = config_filename
        self.tokenizer_name = "camembert-base"
        self.tokenizer = CamembertTokenizer.from_pretrained(self.tokenizer_name)
        self.model = None
        self.config = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.trainer = None
        
        self.base_dir = base_dir
        self.run_dir = self._setup_run_dir() 
        self.saver = FineTuningSaver(run_dir=self.run_dir)

    def _setup_run_dir(self) -> str:
        """Create and setup the run directory with proper error handling."""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            run_id = 0
            while True:
                run_dir = os.path.join(self.base_dir, f"nli_run{run_id}")
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)
                    # Create required subdirectories
                    for subdir in ['checkpoints', 'weights', 'logs', 'metrics']:
                        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
                    return run_dir
                run_id += 1
        except Exception as e:
            logger.error(f"Failed to setup run directory: {e}")
            # Fallback to timestamp-based directory
            import time
            fallback_dir = os.path.join(self.base_dir, f"nli_run_{int(time.time())}")
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir

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

    def finetune_model(self, train_dataset, test_dataset, validation_dataset, 
                    num_epochs=1, batch=8, learning_rate=1e-5, num_labels=3):
        """Fine-tune with proper directory handling and save validation."""
        if not hasattr(self, 'run_dir') or not os.path.exists(self.run_dir):
            self.run_dir = self._setup_run_dir()
            
        try:
            if not all([self.config, self.model]):
                raise ValueError("Model and config must be initialized first")
                
            # Initialize model
            model = CustomNLIModel(self.config, self.model.state_dict(), num_labels)
            
            # Training arguments with updated parameter names
            training_args = TrainingArguments(
                output_dir=self.run_dir,
                eval_strategy="epoch",  # Updated from evaluation_strategy
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
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )
            
            # Initialize trainer without tokenizer parameter
            trainer = CustomNLITrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                saver=self.saver
            )
            
            return trainer, trainer.train()
            
        except Exception as e:
            logger.error(f"Error in finetune_model: {str(e)}")
            raise
        
    def initialize_model(self, model_repo: str, weights_path: str, config_path: str, tokenizer_name: str = "camembert-base") -> str:
        """Initialize model from local files."""
        try:
            # Construction des chemins complets en incluant camembert-training
            base_dir = "camembert-training"
            weights_full_path = os.path.join(base_dir, model_repo, weights_path)
            config_full_path = os.path.join(base_dir, model_repo, config_path)
            
            # Vérification des fichiers
            if not os.path.exists(weights_full_path):
                return f"❌ Fichier de poids non trouvé: {weights_full_path}"
            if not os.path.exists(config_full_path):
                return f"❌ Fichier de configuration non trouvé: {config_full_path}"
                
            # Chargement du tokenizer (toujours depuis camembert-base)
            try:
                self.tokenizer = CamembertTokenizer.from_pretrained(tokenizer_name)
                self.logger.info(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                return f"❌ Erreur lors du chargement du tokenizer: {str(e)}"
            
            # Chargement du modèle local
            try:
                self.config = AutoConfig.from_pretrained(config_full_path)
                self.model = RobertaForMaskedLM.from_pretrained(
                    weights_full_path,
                    config=self.config
                )
                
                self.logger.info(f"Successfully loaded model from {weights_full_path}")
                return "✅ Modèle chargé avec succès"
                
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return f"❌ Erreur lors du chargement du modèle: {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return f"❌ Erreur inattendue: {str(e)}"

    def prepare_dataset(self, dataset_name: str, language: str, test_split: float) -> Tuple[str, pd.DataFrame]:
        """Prepare the dataset and return status and preview."""
        try:
            self.train_dataset, self.test_dataset, self.val_dataset = \
                self.load_data(dataset_name, language, test_split)
            
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
