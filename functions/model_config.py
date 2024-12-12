import logging
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast


@dataclass
class ModelArguments:
    """Arguments par défaut pour la configuration RoBERTa"""
    vocab_size: int = 50265  # Taille par défaut de RoBERTa
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
        self.tokenizer = None
        self.base_tokenizer = None

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
        try:
            # Initialize base tokenizer with explicit error handling
            try:
                if not hasattr(self, 'base_tokenizer') or self.base_tokenizer is None:
                    self.base_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
                    if self.base_tokenizer is None:
                        raise ValueError("Failed to initialize RoBERTa tokenizer")
            except Exception as e:
                return f"❌ Erreur d'initialisation du tokenizer: {str(e)}"

            original_vocab_size = len(self.base_tokenizer)

            # Validation des paramètres
            if hidden_size % num_attention_heads != 0:
                return "❌ Erreur: La dimension des embeddings doit être divisible par le nombre de têtes d'attention"
                
            if hidden_size <= 0 or num_attention_heads <= 0 or num_hidden_layers <= 0:
                return "❌ Erreur: Les dimensions doivent être strictement positives"
                
            if vocab_size < original_vocab_size:
                return f"❌ Erreur: La taille du vocabulaire doit être >= {original_vocab_size}"
                
            if intermediate_size < hidden_size:
                return "❌ Erreur: La taille intermédiaire doit être supérieure à la dimension des embeddings"
                
            if not (0 <= hidden_dropout_prob <= 1 and 0 <= attention_probs_dropout_prob <= 1):
                return "❌ Erreur: Les probabilités de dropout doivent être entre 0 et 1"

            # Create configuration with validated parameters
            config_params = {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_attention_heads': num_attention_heads,
                'num_hidden_layers': num_hidden_layers,
                'intermediate_size': intermediate_size,
                'hidden_dropout_prob': hidden_dropout_prob,
                'attention_probs_dropout_prob': attention_probs_dropout_prob,
                'pad_token_id': self.base_tokenizer.pad_token_id,
                'bos_token_id': self.base_tokenizer.bos_token_id,
                'eos_token_id': self.base_tokenizer.eos_token_id,
            }
            
            self._create_config(**config_params)
            
            # Initialize model with proper embedding handling
            model = self._initialize_model_with_vocab()
            
            # Initialize training configuration if data is ready
            if run_handler is not None and run_handler.data_loader is not None:
                if not run_handler.data_loader.is_ready():
                    return "❌ Erreur: Veuillez d'abord charger le dataset"
                    
                from .train import TrainingConfig
                run_handler.training_config = TrainingConfig(self, run_handler.data_loader)
                self.logger.info("Configuration d'entraînement initialisée avec succès")

            status = (
                "✅ Configuration du modèle initialisée avec succès!\n\n"
                "Paramètres choisis:\n"
                f"- Architecture: {num_hidden_layers} couches, {num_attention_heads} têtes d'attention\n"
                f"- Dimensions: {hidden_size} embeddings, {intermediate_size} intermédiaire\n"
                f"- Vocabulaire: {vocab_size} tokens (original: {original_vocab_size})\n"
                f"- Regularisation: {hidden_dropout_prob} dropout, {attention_probs_dropout_prob} attention dropout"
            )

            return status
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return f"❌ Erreur lors de l'initialisation: {str(e)}"

    def _create_config(self, **kwargs) -> None:
        """Crée la configuration RoBERTa avec les paramètres fournis"""
        try:
            # Update arguments with provided values
            for key, value in kwargs.items():
                if hasattr(self.model_args, key):
                    setattr(self.model_args, key, value)

            # Create RoBERTa configuration
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
                pad_token_id=kwargs.get('pad_token_id'),
                bos_token_id=kwargs.get('bos_token_id'),
                eos_token_id=kwargs.get('eos_token_id')
            )
            
        except Exception as e:
            self.logger.error(f"Erreur création configuration: {e}")
            raise

    def _initialize_model_with_vocab(self) -> RobertaForMaskedLM:
        try:
            # Détection de l'environnement
            cuda_available = torch.cuda.is_available()
            dtype = torch.float16 if cuda_available else torch.float32
            
            # Initialisation du modèle avec les paramètres appropriés
            model = RobertaForMaskedLM(self.config)
            
            with torch.no_grad():
                # Chargement du modèle de base
                base_model = RobertaForMaskedLM.from_pretrained(
                    "roberta-base",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                
                # Récupération des embeddings originaux
                original_embeddings = base_model.get_input_embeddings().weight.data
                
                # Si les dimensions sont différentes, on fait un redimensionnement
                if original_embeddings.size(1) != self.config.hidden_size:
                    # Redimensionnement linéaire des embeddings pour correspondre à la nouvelle dimension
                    original_embeddings = torch.nn.functional.interpolate(
                        original_embeddings.unsqueeze(0),  # Ajoute une dimension pour le batch
                        size=(self.config.hidden_size,),
                        mode='linear'
                    ).squeeze(0)  # Retire la dimension de batch
                
                # Initialisation des embeddings
                new_embeddings = model.get_input_embeddings()
                new_embeddings.weight.data[:len(self.base_tokenizer)] = original_embeddings
                
                if self.config.vocab_size > len(self.base_tokenizer):
                    size = (self.config.vocab_size - len(self.base_tokenizer), self.config.hidden_size)
                    
                    # Calcul des statistiques pour l'initialisation
                    token_norms = torch.norm(original_embeddings, dim=1)
                    mean = token_norms.mean()
                    std = token_norms.std()
                    
                    # Initialisation des nouveaux tokens
                    new_tokens = torch.normal(mean=mean, std=std, size=size, dtype=dtype)
                    
                    # Mise à jour des embeddings
                    new_embeddings.weight.data[len(self.base_tokenizer):] = new_tokens
                    model.get_output_embeddings().weight.data[:len(self.base_tokenizer)] = original_embeddings
                    model.get_output_embeddings().weight.data[len(self.base_tokenizer):] = new_tokens
                
                # Nettoyage
                del base_model
                if cuda_available:
                    torch.cuda.empty_cache()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error initializing model with vocabulary: {e}")
            raise

    def get_config(self) -> Optional[RobertaConfig]:
        """Récupère la configuration actuelle"""
        return self.config