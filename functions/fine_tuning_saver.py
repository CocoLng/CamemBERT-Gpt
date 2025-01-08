import logging
import os
from typing import Dict, Optional, Tuple

import torch
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

from .training_saver import TrainingSaver


class FineTuningSaver(TrainingSaver):
    """Handles saving and loading operations for NLI fine-tuning."""

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        
        # Configuration des chemins spécifiques au NLI
        self.run_dir = kwargs.pop("run_dir", "camembert-nli")
        self.weights_dir = os.path.join(self.run_dir, "weights")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")

        # Création des dossiers nécessaires
        for directory in [self.weights_dir, self.checkpoints_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialisation de la classe parente
        super().__init__(*args, run_dir=self.run_dir, **kwargs)

    def save_nli_model_info(self, config: Dict, dataset_info: Dict) -> None:
        """Save NLI specific model information."""
        try:
            info_path = os.path.join(self.run_dir, "nli_model_info.txt")
            with open(info_path, "w") as f:
                f.write("=== NLI Model Configuration ===\n")
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n=== Dataset Information ===\n")
                for key, value in dataset_info.items():
                    f.write(f"{key}: {value}\n")

            self.logger.info(f"NLI model info saved to {info_path}")
        except Exception as e:
            self.logger.error(f"Error saving NLI model info: {e}")

    def save_evaluation_metrics(self, results: Dict) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:
        """Save and visualize evaluation metrics."""
        try:
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                "Metric": list(results.keys()),
                "Value": list(results.values())
            })
            
            # Save metrics to CSV
            metrics_path = os.path.join(self.metrics_dir, "evaluation_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)

            # Create and save confusion matrix if available
            cm_fig = None
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                cm_fig = plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Matrice de Confusion')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(self.metrics_dir, "confusion_matrix.png"))
                plt.close()

            # Create and save learning curves if available
            acc_fig = None
            if 'history' in results:
                history = results['history']
                acc_fig = plt.figure(figsize=(10, 6))
                if 'train_loss' in history:
                    plt.plot(history['train_loss'], label='Train Loss')
                if 'val_loss' in history:
                    plt.plot(history['val_loss'], label='Validation Loss')
                plt.title('Courbe d\'Apprentissage')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(self.metrics_dir, "learning_curves.png"))
                plt.close()

            return metrics_df, cm_fig, acc_fig

        except Exception as e:
            self.logger.error(f"Error saving evaluation metrics: {e}")
            return pd.DataFrame(), None, None

    def save_nli_checkpoint(self, step: int, metrics: Optional[Dict] = None) -> None:
        """Save NLI specific checkpoint."""
        try:
            checkpoint_path = os.path.join(self.checkpoints_dir, f"nli_checkpoint-{step}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save the model state
            if hasattr(self, "model") and self.model is not None:
                self.model.save_pretrained(checkpoint_path)
                if hasattr(self, "tokenizer"):
                    self.tokenizer.save_pretrained(checkpoint_path)

            # Save training state
            if hasattr(self, "state"):
                training_state = {
                    "step": step,
                    "metrics": metrics or {},
                    "log_history": getattr(self.state, "log_history", [])
                }
                torch.save(training_state, os.path.join(checkpoint_path, "nli_trainer_state.pt"))

            self.logger.info(f"NLI checkpoint saved to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error saving NLI checkpoint: {e}")
            raise

    def load_nli_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load NLI checkpoint and return state information."""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

            # Load training state
            state_path = os.path.join(checkpoint_path, "nli_trainer_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path)
            else:
                state = {}

            return {
                "path": checkpoint_path,
                "state": state,
                "status": "Checkpoint chargé avec succès"
            }
        except Exception as e:
            self.logger.error(f"Error loading NLI checkpoint: {e}")
            return {
                "path": checkpoint_path,
                "state": {},
                "status": f"Erreur: {str(e)}"
            }

    def get_available_checkpoints(self) -> List[str]:
        """Get list of available NLI checkpoints."""
        try:
            checkpoints = []
            for item in os.listdir(self.checkpoints_dir):
                if item.startswith("nli_checkpoint-"):
                    checkpoints.append(item)
            return sorted(checkpoints)
        except Exception as e:
            self.logger.error(f"Error getting available checkpoints: {e}")
            return []

    def cleanup_old_checkpoints(self, max_checkpoints: int = 5) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        try:
            checkpoints = self.get_available_checkpoints()
            if len(checkpoints) > max_checkpoints:
                checkpoints_to_remove = checkpoints[:-max_checkpoints]
                for checkpoint in checkpoints_to_remove:
                    checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint)
                    if os.path.exists(checkpoint_path):
                        for file in os.listdir(checkpoint_path):
                            os.remove(os.path.join(checkpoint_path, file))
                        os.rmdir(checkpoint_path)
                        self.logger.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old checkpoints: {e}")
