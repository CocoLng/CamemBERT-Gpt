import logging
from pathlib import Path
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import itertools
import multiprocessing

class FrenchTokenizerTrainer:
    """Gestionnaire d'entraînement du tokenizer pour le français"""
    
    def __init__(self, vocab_size: int = 32000):
        self.logger = logging.getLogger(__name__)
        self.vocab_size = vocab_size
        
    def prepare_training_data(self, output_path: str):
        """Prépare les données pour l'entraînement du tokenizer"""
        try:
            output_file = Path(output_path) / "tokenizer_training_data.txt"
            if output_file.exists():
                self.logger.info("Fichier de données d'entraînement existant trouvé. Chargement direct.")
                return str(output_file)
            
            # Charge le dataset OSCAR français en streaming
            dataset = load_dataset(
                "oscar",
                "unshuffled_deduplicated_fr",
                streaming=True,
                split="train",
                trust_remote_code=True
            )
            
            # Échantillonne 10^7 phrases aléatoirement
            self.logger.info("Sampling 10M sentences for tokenizer training...")
            
            def is_valid(example):
                return isinstance(example["text"], str)
            
            sampled_texts = itertools.islice((ex["text"] for ex in dataset if is_valid(ex)), 10_000_000)
            
            # Écriture directe sans stocker en mémoire avec un buffer plus grand
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with output_file.open("w", encoding="utf-8", buffering=1024*1024) as f:  # Buffer de 1MB
                for text in sampled_texts:
                    f.write(text + "\n")
                    
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise

    def train_tokenizer(self, input_file: str, output_path: str):
        try:
            model_prefix = str(Path(output_path) / "french_sp")
            
            cpu_count = multiprocessing.cpu_count()
            reduced_threads = max(1, cpu_count // 2)  # Réduire le nombre de threads à la moitié
            
            # Configuration améliorée de SentencePiece pour CamemBERT
            spm.SentencePieceTrainer.train(
                input=input_file,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                character_coverage=0.99999,  # Couverture caractères très élevée pour le français
                model_type="unigram",        # Comme spécifié dans l'article
                add_dummy_prefix=False,      # Pas de préfixe fictif
                normalization_rule_name="identity",  # Pas de normalisation
                user_defined_symbols=["<s>", "</s>", "<pad>", "<mask>"], 
                # Ajouts recommandés:
                input_sentence_size=10000000,  # Pour les 10^7 phrases
                shuffle_input_sentence=True,   # Mélange des phrases
                train_extremely_large_corpus=True,  # Pour OSCAR
                num_threads=reduced_threads  # Utiliser la moitié des cœurs disponibles
            )
            
            return model_prefix + ".model", model_prefix + ".vocab"
            
        except Exception as e:
            self.logger.error(f"Error training tokenizer: {e}")
            raise
            
    def load_tokenizer(self, model_path: str) -> PreTrainedTokenizerFast:
        """Charge le tokenizer entraîné en tant que PreTrainedTokenizerFast"""
        try:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=model_path,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                padding_side="right",
                truncation_side="right",
                model_max_length=512
            )
            
            # Configure les tokens spéciaux
            special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>"
            }
            tokenizer.add_special_tokens(special_tokens)
            
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise