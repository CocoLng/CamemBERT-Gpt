import logging
from typing import Dict, List
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb

class MaskingMonitorCallback(TrainerCallback):
    """Callback amélioré pour monitorer le masquage pendant l'entraînement"""
    
    def __init__(self, tokenizer, expected_mlm_probability: float = 0.15,
                 alert_threshold: float = 0.05, rolling_window: int = 100):
        self.logger = logging.getLogger(__name__)
        self.mask_token_id = tokenizer.mask_token_id
        self.expected_ratio = expected_mlm_probability
        self.alert_threshold = alert_threshold
        self.rolling_window = rolling_window
        self.check_frequency = 100
        
        # Statistiques de masquage
        self.total_masks = 0
        self.total_tokens = 0
        self.current_batch_ratio = 0.0
        self.overall_ratio = 0.0
        self.history: List[float] = []
        self.anomaly_count = 0
        
        self._current_batch_stats = None
        
    def analyze_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyse détaillée du ratio de masquage dans un batch"""
        try:
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            
            # Calcul des tokens valides et masqués
            valid_tokens = attention_mask.sum().item()
            masked_tokens = (input_ids == self.mask_token_id).sum().item()
            
            # Mise à jour des statistiques
            self.total_masks += masked_tokens
            self.total_tokens += valid_tokens
            self.current_batch_ratio = masked_tokens / valid_tokens if valid_tokens > 0 else 0
            self.overall_ratio = self.total_masks / self.total_tokens if self.total_tokens > 0 else 0
            
            # Mise à jour de l'historique pour la moyenne mobile
            self.history.append(self.current_batch_ratio)
            if len(self.history) > self.rolling_window:
                self.history.pop(0)
            
            # Calcul des statistiques courantes
            self._current_batch_stats = {
                'current_masking_ratio': self.current_batch_ratio,  # Clé modifiée
                'overall_masking_ratio': self.overall_ratio,  # Clé modifiée
                'expected_ratio': self.expected_ratio,
                'rolling_average': self.get_rolling_average(),
                'deviation_from_expected': abs(self.get_rolling_average() - self.expected_ratio),
                'total_masks': self.total_masks,
                'total_tokens': self.total_tokens
            }
            
            # Vérification des anomalies
            if self._check_for_anomalies(self._current_batch_stats):
                self._handle_anomaly(self._current_batch_stats)
            
            return self._current_batch_stats
            
        except Exception as e:
            self.logger.error(f"Erreur dans l'analyse du batch: {e}")
            return None
            
    def get_stats_dict(self) -> Dict[str, float]:
        """Retourne les statistiques courantes de masquage"""
        if self._current_batch_stats is None:
            return {
                'current_masking_ratio': 0.0,  # Clé modifiée
                'overall_masking_ratio': 0.0,  # Clé modifiée
                'expected_ratio': self.expected_ratio,
                'rolling_average': 0.0,
                'deviation_from_expected': 0.0,
                'total_masks': 0,
                'total_tokens': 0
            }
        return self._current_batch_stats
    
    def get_rolling_average(self) -> float:
        """Calcule la moyenne mobile du ratio de masquage"""
        return sum(self.history) / len(self.history) if self.history else 0
    
    def _check_for_anomalies(self, stats: Dict[str, float]) -> bool:
        """Détecte les anomalies dans les statistiques de masquage"""
        return stats["deviation_from_expected"] > self.alert_threshold
    
    def _handle_anomaly(self, stats: Dict[str, float]) -> None:
        """Gestion des anomalies de masquage"""
        self.anomaly_count += 1
        self.logger.warning(
            f"Anomalie de masquage détectée (#{self.anomaly_count}):\n"
            f"- Ratio actuel: {stats['current_masking_ratio']:.2%}\n"  # Clé modifiée
            f"- Moyenne mobile: {stats['rolling_average']:.2%}\n"
            f"- Déviation: {stats['deviation_from_expected']:.2%}\n"
            f"- Seuil: {self.alert_threshold:.2%}"
        )
    
    def reset_stats(self) -> None:
        """Réinitialise toutes les statistiques de masquage"""
        self.total_masks = 0
        self.total_tokens = 0
        self.current_batch_ratio = 0.0
        self.overall_ratio = 0.0
        self.history.clear()
        self.anomaly_count = 0
        self._current_batch_stats = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Initialisation et logging des paramètres de monitoring"""
        self.logger.info(
            f"Démarrage du monitoring de masquage:\n"
            f"- Ratio attendu: {self.expected_ratio:.2%}\n"
            f"- Seuil d'alerte: {self.alert_threshold:.2%}\n"
            f"- Fenêtre mobile: {self.rolling_window} batchs\n"
            f"- Fréquence de vérification: {self.check_frequency} steps"
        )
        
        # Initialiser le tracking W&B si disponible
        if wandb.run is not None:
            wandb.define_metric("masking/current_masking_ratio")  # Clé modifiée
            wandb.define_metric("masking/rolling_average")
            wandb.define_metric("masking/deviation")

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, **kwargs):
        """Monitoring à chaque étape d'entraînement"""
        if state.global_step % self.check_frequency == 0:
            stats = self.get_stats_dict()
            
            # Logging vers W&B si disponible
            if wandb.run is not None:
                wandb.log({
                    "masking/current_masking_ratio": stats["current_masking_ratio"],  # Clé modifiée
                    "masking/rolling_average": stats["rolling_average"],
                    "masking/deviation": stats["deviation_from_expected"]
                }, step=state.global_step)