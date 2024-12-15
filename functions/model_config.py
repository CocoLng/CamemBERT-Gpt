import logging
from dataclasses import dataclass, field

from transformers import RobertaConfig, RobertaTokenizerFast


@dataclass
class ModelArguments:
    """Arguments par défaut pour la configuration RoBERTa améliorée"""

    # Paramètres 
    vocab_size: int = 50265
    max_position_embeddings: int = 514
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"  # Activation par défaut RoBERTa

    learning_rate: float = 6e-4  # Learning rate de base CamemBERT
    warmup_ratio: float = 0.06  # 6% comme CamemBERT
    weight_decay: float = 0.01  # Valeur par défaut AdamW
    adam_epsilon: float = 1e-6  # Epsilon pour Adam
    max_grad_norm: float = 1.0  # Gradient clipping
    batch_size: int = 8192  # Taille batch recommandée

    # Paramètres tokenizer
    special_tokens: dict = field(
        default_factory=lambda: {
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        }
    )


class ModelConfig:
    """Gestionnaire de configuration du modèle RoBERTa"""

    def __init__(self):
        """Initialisation du gestionnaire de configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.model_args = ModelArguments()
        self.tokenizer = None
        self.base_tokenizer = None

    def initialize_full_config(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        run_handler=None,
    ) -> tuple[str, str]:
        try:
            try:
                # Vérifier si le tokenizer est initialisé
                if not hasattr(self, "base_tokenizer") or self.base_tokenizer is None:
                    self.base_tokenizer = RobertaTokenizerFast.from_pretrained(
                        "roberta-base"
                    )
                    if self.base_tokenizer is None:
                        raise ValueError("Failed to initialize RoBERTa tokenizer")
            except Exception as e:
                return f"❌ Erreur d'initialisation du tokenizer: {str(e)}", ""

            # Vérifier si le vocabulaire a été chargé
            original_vocab_size = len(self.base_tokenizer)
            vocab_size = self.model_args.vocab_size

            # Validation des paramètres
            if hidden_size % num_attention_heads != 0:
                return (
                    "❌ Erreur: La dimension des embeddings doit être divisible par le nombre de têtes d'attention",
                    "",
                )

            if hidden_size <= 0 or num_attention_heads <= 0 or num_hidden_layers <= 0:
                return (
                    "❌ Erreur: Les dimensions doivent être strictement positives",
                    "",
                )

            if vocab_size < original_vocab_size:
                return (
                    f"❌ Erreur: La taille du vocabulaire doit être >= {original_vocab_size}",
                    "",
                )

            if intermediate_size < hidden_size:
                return (
                    "❌ Erreur: La taille intermédiaire doit être supérieure à la dimension des embeddings",
                    "",
                )

            if not (
                0 <= hidden_dropout_prob <= 1 and 0 <= attention_probs_dropout_prob <= 1
            ):
                return (
                    "❌ Erreur: Les probabilités de dropout doivent être entre 0 et 1",
                    "",
                )

            # Paramètres de configuration
            config_params = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "intermediate_size": intermediate_size,
                "hidden_dropout_prob": hidden_dropout_prob,
                "attention_probs_dropout_prob": attention_probs_dropout_prob,
                "pad_token_id": self.base_tokenizer.pad_token_id,
                "bos_token_id": self.base_tokenizer.bos_token_id,
                "eos_token_id": self.base_tokenizer.eos_token_id,
            }

            self._create_config(**config_params)

            # Initialisation du tokenizer
            if run_handler is not None and run_handler.data_loader is not None:
                if not run_handler.data_loader.is_ready():
                    return "❌ Erreur: Veuillez d'abord charger le dataset", ""

                from .train import TrainingConfig

                run_handler.training_config = TrainingConfig(
                    self, run_handler.data_loader
                )

                # Calculer automatiquement les paramètres optimaux
                training_params = (
                    run_handler.training_config._calculate_training_parameters()
                )

                # Préparer les messages de statut
                model_status = (
                    "✅ Configuration du modèle initialisée avec succès!\n\n"
                    f"Paramètres du modèle:\n"
                    f"- Architecture: {num_hidden_layers} couches, {num_attention_heads} têtes d'attention\n"
                    f"- Dimensions: {hidden_size} embeddings, {intermediate_size} intermédiaire\n"
                    f"- Vocabulaire: {vocab_size} tokens (original: {original_vocab_size})\n"
                    f"- Regularisation: {hidden_dropout_prob} dropout, {attention_probs_dropout_prob} attention dropout"
                )

                return model_status, training_params["log_message"]
            else:
                return (
                    "✅ Configuration du modèle initialisée avec succès, "
                    "mais pas de dataset chargé pour l'entraînement",
                    "",
                )

        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return f"❌ Erreur lors de l'initialisation: {str(e)}", ""

    def _create_config(self, **kwargs) -> None:
        """Crée la configuration RoBERTa avec les paramètres fournis"""
        try:
            # Mise à jour des paramètres du modèle demandés par l'interface
            for key, value in kwargs.items():
                if hasattr(self.model_args, key):
                    setattr(self.model_args, key, value)

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
                layer_norm_eps=self.model_args.layer_norm_eps,
                pad_token_id=kwargs.get("pad_token_id"),
                bos_token_id=kwargs.get("bos_token_id"),
                eos_token_id=kwargs.get("eos_token_id"),
            )

        except Exception as e:
            self.logger.error(f"Erreur création configuration: {e}")
            raise
