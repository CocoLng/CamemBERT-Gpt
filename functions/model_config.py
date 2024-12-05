import logging
from dataclasses import dataclass
from typing import Optional

from transformers import RobertaConfig


@dataclass
class ModelArguments:
    """Arguments par défaut pour la configuration RoBERTa"""
    vocab_size: int = 50265
    max_position_embeddings: int = 514
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5


class ModelConfig:
    """Gestionnaire de configuration du modèle RoBERTa"""
    
    def __init__(self):
        """Initialisation du gestionnaire de configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.model_args = ModelArguments()

    def initialize_full_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        run_handler = None
    ) -> str:
        """
        Initialise la configuration complète du modèle avec validation
        Retourne un message formaté pour l'interface
        """
        try:
            # Validation des dimensions
            if hidden_size % num_attention_heads != 0:
                return "❌ Erreur: La dimension des embeddings doit être divisible par le nombre de têtes d'attention"

            # Mise à jour de la configuration
            config_params = {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_attention_heads': num_attention_heads,
                'num_hidden_layers': num_hidden_layers,
                'intermediate_size': intermediate_size,
                'hidden_dropout_prob': hidden_dropout_prob,
                'attention_probs_dropout_prob': attention_probs_dropout_prob
            }
            
            self._create_config(**config_params)

            # Initialisation de la configuration d'entraînement si les données sont prêtes
            if run_handler is not None and run_handler.data_loader is not None:
                if not run_handler.data_loader.is_ready():
                    return "❌ Erreur: Veuillez d'abord charger le dataset"
                    
                from .train import TrainingConfig
                run_handler.training_config = TrainingConfig(self, run_handler.data_loader)
                self.logger.info("Configuration d'entraînement initialisée avec succès")

            # Création du message de statut
            status = (
                "✅ Configuration du modèle initialisée avec succès!\n\n"
                "Paramètres choisis:\n"
                f"- Architecture: {num_hidden_layers} couches, {num_attention_heads} têtes d'attention\n"
                f"- Dimensions: {hidden_size} embeddings, {intermediate_size} intermédiaire\n"
                f"- Vocabulaire: {vocab_size} tokens\n"
                f"- Regularisation: {hidden_dropout_prob} dropout, {attention_probs_dropout_prob} attention dropout"
            )

            return status
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return f"❌ Erreur lors de l'initialisation: {str(e)}"

    def _create_config(self, **kwargs) -> None:
        """Crée la configuration RoBERTa avec les paramètres fournis"""
        try:
            # Mise à jour des arguments avec les valeurs fournies
            for key, value in kwargs.items():
                if hasattr(self.model_args, key):
                    setattr(self.model_args, key, value)

            # Création de la configuration RoBERTa
            self.config = RobertaConfig(
                vocab_size=self.model_args.vocab_size,
                max_position_embeddings=self.model_args.max_position_embeddings,
                hidden_size=self.model_args.hidden_size,
                num_attention_heads=self.model_args.num_attention_heads,
                num_hidden_layers=self.model_args.num_hidden_layers,
                intermediate_size=self.model_args.intermediate_size,
                hidden_act=self.model_args.hidden_act,
                hidden_dropout_prob=self.model_args.hidden_dropout_prob,
                attention_probs_dropout_prob=self.model_args.attention_probs_dropout_prob,
                type_vocab_size=self.model_args.type_vocab_size,
                layer_norm_eps=self.model_args.layer_norm_eps
            )
            
        except Exception as e:
            self.logger.error(f"Erreur création configuration: {e}")
            raise

    def get_config(self) -> Optional[RobertaConfig]:
        """Récupère la configuration actuelle"""
        return self.config