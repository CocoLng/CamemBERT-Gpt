import logging
import os
from typing import Optional

import torch
import wandb


class TrainingSaver:
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Configuration des chemins
        self.run_dir = kwargs.pop("run_dir", "camembert-training")
        self.weights_dir = os.path.join(self.run_dir, "weights")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")

        # Création des dossiers nécessaires
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Attributs de base
        self.dataset_size = kwargs.pop("dataset_size", None)
        self.tokens_processed = 0
        self.processing_class = kwargs.pop("processing_class", None)

    def _save_model_info(self, directory: str):
        """Sauvegarde les informations complètes du modèle."""
        try:
            if not hasattr(self, "model") or self.model is None:
                self.logger.warning(
                    "Model not initialized yet, skipping model info save"
                )
                return

            info_path = os.path.join(directory, "model_info.txt")
            with open(info_path, "w") as f:
                f.write("=== Dataset Information ===\n")
                if self.dataset_size:
                    f.write(f"Total tokens: {self.dataset_size:,}\n")
                    f.write(
                        f"Estimated size: {self.dataset_size * 4 / (1024**3):.2f} GB\n"
                    )
                f.write("\n")

                f.write("=== Model Architecture ===\n")
                config_dict = self.model.config.to_dict()
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                f.write("=== Training Parameters ===\n")
                if hasattr(self, "args"):
                    training_params = {
                        "learning_rate": self.args.learning_rate,
                        "batch_size": self.args.per_device_train_batch_size,
                        "gradient_accumulation": self.args.gradient_accumulation_steps,
                        "warmup_steps": self.args.warmup_steps,
                        "total_steps": self.args.max_steps,
                    }
                    for param, value in training_params.items():
                        f.write(f"{param}: {value}\n")

                f.write("\n=== Dataset Configuration ===\n")
                if self.data_loader and self.data_loader.dataset_config:
                    dataset_config = self.data_loader.dataset_config
                    f.write(f"Dataset: {dataset_config.name}\n")
                    f.write(f"Subset: {dataset_config.subset}\n")
                    f.write(f"Split: {dataset_config.split}\n")
                    f.write(f"Streaming: {dataset_config.streaming}\n")

        except Exception as e:
            self.logger.error(f"Error saving model info: {e}")
            return

    def _save_metrics_report(self, checkpoint_dir: str):
        """Sauvegarde un rapport détaillé des métriques."""
        try:
            report_path = os.path.join(checkpoint_dir, "metrics_report.txt")
            with open(report_path, "w") as f:
                f.write("=== Training Metrics Report ===\n\n")

                # Current state
                f.write("Current State:\n")
                f.write(f"Step: {self.state.global_step}\n")
                f.write(f"Tokens Processed: {self.tokens_processed:,}\n")
                if self.dataset_size:
                    progress = self.tokens_processed / self.dataset_size
                    f.write(f"Training Progress: {progress:.2%}\n")

                f.write("\nRecent Metrics:\n")
                if self.state.log_history:
                    recent_logs = self.state.log_history[-5:]
                    for log in recent_logs:
                        f.write(f"Step {log.get('step', 'N/A')}:\n")
                        for key, value in log.items():
                            if key != "step":
                                f.write(f"  {key}: {value}\n")
                        f.write("\n")

                if self.state.best_model_checkpoint:
                    f.write(f"\nBest Checkpoint: {self.state.best_model_checkpoint}\n")
                    if hasattr(self.state, "best_metric"):
                        f.write(f"Best Metric: {self.state.best_metric}\n")

        except Exception as e:
            self.logger.error(f"Error saving metrics report: {e}")

    def _save_final_metrics(self, metrics_path: str):
        """Sauvegarde les métriques finales d'entraînement."""
        final_metrics = {
            "total_steps": self.state.global_step,
            "total_tokens": self.tokens_processed,
            "training_time": self.state.total_flos,
            "final_loss": self.state.log_history[-1].get("loss")
            if self.state.log_history
            else None,
            "best_model_checkpoint": self.state.best_model_checkpoint,
        }

        if hasattr(self.state, "best_metric"):
            final_metrics["best_metric"] = self.state.best_metric

        torch.save(final_metrics, metrics_path)

    def save_model(self, output_dir: Optional[str] = None):
        """Sauvegarde le modèle final, le tokenizer et les informations associées."""
        try:
            save_dir = output_dir if output_dir else self.weights_dir
            os.makedirs(save_dir, exist_ok=True)

            # Sauvegarde du modèle
            self.logger.info("Saving final model...")
            self.model.save_pretrained(save_dir, safe_serialization=True)

            # Sauvegarde du tokenizer ou de la classe de traitement
            if hasattr(self, "tokenizer"):
                self.logger.info("Saving tokenizer...")
                self.tokenizer.save_pretrained(save_dir)
            elif self.processing_class is not None:
                self.logger.info("Saving processing class...")
                self.processing_class.save_pretrained(save_dir)
            else:
                self.logger.warning("No tokenizer or processing class found to save")

            # Sauvegarder les informations du modèle
            self._save_model_info(self.run_dir)

            # Sauvegarde des métriques finales
            metrics_path = os.path.join(save_dir, "final_metrics.json")
            self._save_final_metrics(metrics_path)
                
            self.logger.info(f"Final model and tokenizer saved in {save_dir}")

            # Synchronisation avec wandb si actif
            if wandb.run:
                self.logger.info("Syncing with wandb...")
                wandb.save(os.path.join(save_dir, "*"))

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            if wandb.run:
                wandb.finish()
            raise

    def save_checkpoint(self, step, metrics=None):
        """Sauvegarde un checkpoint du modèle et de l'état d'entraînement."""
        try:
            checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Sauvegarder le modèle et le tokenizer
            self.model.save_pretrained(checkpoint_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(checkpoint_path)
                self.logger.info(f"Tokenizer saved to {checkpoint_path}")

            # Sauvegarder l'état d'entraînement
            training_state = {
                "global_step": self.state.global_step,
                "tokens_processed": self.tokens_processed,
                "target_size": self.dataset_size,
                "log_history": self.state.log_history,
                "best_model_checkpoint": self.state.best_model_checkpoint,
                "training_time": self.state.total_flos,
            }

            # Ajouter les métriques si présentes
            if metrics:
                training_state["metrics"] = metrics

            torch.save(
                training_state, os.path.join(checkpoint_path, "trainer_state.pt")
            )

            # Sauvegarder les métriques
            self._save_metrics_report(checkpoint_path)

            self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise
