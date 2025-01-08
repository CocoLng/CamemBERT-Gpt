import logging
from typing import Dict, List, Tuple, Set
import torch
import re

class TestPredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Tokens spéciaux communs à RoBERTa et CamemBERT
        self.special_tokens = {
            '<s>', '</s>', '<pad>', '<unk>', '<mask>',
            # Tokens RoBERTa spécifiques
            '<additional_special_tokens>', 
            # Tokens CamemBERT spécifiques
            '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'
        }
        
        # Conversion des tokens spéciaux en IDs pour les deux tokenizers
        self.special_token_ids = self._get_all_special_token_ids()
        
        # Liste des tokens de bruit fréquents
        self.noise_tokens = {
            'fre', 'rolex', 'beach', '4k', 'video', 'pro',
            'download', 'upload', 'click', 'subscribe'
        }
        
        # Regex pour la validation des tokens français
        self.french_token_pattern = re.compile(
            r'^[a-zà-ÿœæ]+$|'  # Mots français minuscules
            r'^[A-ZÀ-ŸŒÆ][a-zà-ÿœæ]*$|'  # Mots avec majuscule initiale
            r'^[,.!?;:]+$'  # Ponctuation
        )
        
        # Cache pour les tokens valides
        self.valid_tokens_cache: Dict[str, bool] = {}
        
        # Règles grammaticales françaises basiques
        self.grammar_rules = {
            'je': {'suis', 'vais', 'peux', 'dois', 'me', 'vous', 'te', 'la', 'le'},
            'tu': {'es', 'vas', 'peux', 'dois', 'me', 'te', 'la', 'le'},
            'il': {'est', 'va', 'peut', 'doit', 'me', 'te', 'la', 'le'},
            'nous': {'sommes', 'allons', 'pouvons', 'devons'},
            'vous': {'êtes', 'allez', 'pouvez', 'devez'},
            'ils': {'sont', 'vont', 'peuvent', 'doivent'}
        }
        
    def _get_all_special_token_ids(self) -> Set[int]:
        """Récupère tous les IDs des tokens spéciaux"""
        special_ids = set()
        for token in self.special_tokens:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id:
                    special_ids.add(token_id)
            except Exception as e:
                self.logger.debug(f"Token spécial non trouvé: {token}, {str(e)}")
        return special_ids

    def clean_token(self, token: str) -> str:
        """Nettoie et normalise un token"""
        if token is None:
            return ""
        # Gestion du caractère spécial de début de mot de CamemBERT
        token = token.replace('▁', ' ')
        # Nettoyage des espaces
        token = token.strip()
        return token

    def is_valid_french_token(self, token: str) -> bool:
        """Vérifie si un token est un mot français valide"""
        if token in self.valid_tokens_cache:
            return self.valid_tokens_cache[token]
        
        token = self.clean_token(token)
        
        # Règles de validation
        is_valid = (
            len(token) >= 2 and  # Longueur minimum
            self.french_token_pattern.match(token) and  # Format valide
            not any(noise in token.lower() for noise in self.noise_tokens)  # Pas de bruit
        )
        
        self.valid_tokens_cache[token] = is_valid
        return is_valid

    def check_grammar_consistency(self, prev_tokens: List[str], current_token: str) -> bool:
        """Vérifie la cohérence grammaticale basique"""
        if not prev_tokens:
            return True
            
        last_token = prev_tokens[-1].lower()
        current = current_token.lower()
        
        # Vérification des règles grammaticales
        if last_token in self.grammar_rules:
            return current in self.grammar_rules[last_token]
            
        return True

    def get_top_predictions(
        self,
        input_ids: torch.Tensor,
        token_index: int,
        top_k: int = 5,
        previous_predictions: List[str] = None,
        temperature: float = 0.7
    ) -> List[Dict]:
        """Obtient les meilleures prédictions avec filtrage amélioré"""
        previous_predictions = previous_predictions or []

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            # Application de la température
            logits = logits / temperature
            
            # Masquage des tokens spéciaux
            for token_id in self.special_token_ids:
                logits[0, token_index, token_id] = float('-inf')
            
            # Calcul des probabilités
            probs = torch.nn.functional.softmax(logits[0, token_index], dim=-1)
            
            # Récupération d'un plus grand nombre de candidats pour le filtrage
            top_probs, top_indices = torch.topk(probs, min(top_k * 15, len(probs)))
            
            predictions = []
            seen_tokens = set()
            
            for prob, idx in zip(top_probs, top_indices):
                if len(predictions) >= top_k:
                    break
                
                token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                cleaned_token = self.clean_token(token)
                
                # Vérifications multiples
                if (not cleaned_token or
                    cleaned_token in seen_tokens or
                    cleaned_token in previous_predictions or
                    not self.is_valid_french_token(cleaned_token) or
                    not self.check_grammar_consistency(previous_predictions, cleaned_token)):
                    continue
                
                # Ajustement de la probabilité pour favoriser les tokens plus longs
                adjusted_prob = prob * (1 + 0.1 * min(len(cleaned_token), 5))
                
                seen_tokens.add(cleaned_token)
                predictions.append({
                    "token": cleaned_token,
                    "token_id": int(idx),
                    "probability": float(adjusted_prob) * 100
                })
            
            return predictions

    def predict_next_tokens(
        self,
        text: str,
        num_tokens: int = 5,
        top_k: int = 5
    ) -> Tuple[List[Dict], str]:
        """Prédit plusieurs tokens de manière séquentielle avec gestion du contexte"""
        try:
            self.logger.info(f"\n=== Starting prediction for text: {text} ===")
            
            text = text.strip()
            current_text = text
            all_predictions = []
            generated_tokens = []
            
            # Tokenisation initiale
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.model.cuda()
            
            input_ids = inputs["input_ids"]
            
            for i in range(num_tokens):
                # Ajustement de la température en fonction de la position
                temperature = max(0.6, min(0.9, 0.6 + i * 0.05))
                
                predictions = self.get_top_predictions(
                    input_ids,
                    token_index=len(input_ids[0]) - 1,
                    top_k=top_k,
                    previous_predictions=generated_tokens,
                    temperature=temperature
                )
                
                if not predictions:
                    self.logger.warning(f"No valid predictions at position {i+1}")
                    break
                
                all_predictions.append({
                    "position": i + 1,
                    "context": current_text,
                    "predictions": predictions
                })
                
                # Mise à jour du texte
                next_token = predictions[0]["token"]
                generated_tokens.append(next_token)
                current_text = f"{current_text}{' ' if not current_text.endswith(' ') else ''}{next_token}"
                
                # Nouvelle tokenisation pour le prochain token
                inputs = self.tokenizer(
                    current_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                input_ids = inputs["input_ids"]
            
            return all_predictions, current_text

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise

    def format_predictions_for_display(self, original_text: str, predictions: List[Dict]) -> str:
        """Formate les prédictions pour l'affichage dans Gradio"""
        formatted = "📝 Détails des prédictions :\n\n"
        
        for pred_group in predictions:
            position = pred_group['position']
            context = pred_group['context']
            
            formatted += f"Position {position} (Contexte: {context}):\n\n```\n"
            
            for pred in pred_group["predictions"]:
                prob = pred['probability']
                token = pred['token']
                
                bar_length = int(prob / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                prob_str = f"{prob:.1f}%".rjust(6)
                
                formatted += f"  {prob_str} |{bar}| '{token}'\n"
            
            formatted += "```\n\n"
        
        return formatted

    def predict_and_display(self, text: str, num_tokens: int, top_k: int) -> Tuple[str, str]:
        """Point d'entrée principal pour les prédictions"""
        try:
            if not self.model:
                return "Erreur: Modèle non initialisé", "Veuillez d'abord configurer et entraîner le modèle"

            predictions, generated_text = self.predict_next_tokens(
                text,
                num_tokens=int(num_tokens),
                top_k=int(top_k)
            )

            formatted_predictions = self.format_predictions_for_display(text, predictions)
            final_text = f"Texte original : {text}\nTexte généré : {generated_text}"
            return final_text, formatted_predictions

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return f"Erreur: {str(e)}", "Une erreur est survenue lors de la prédiction"