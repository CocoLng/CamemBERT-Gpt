import logging
from typing import Dict, List, Tuple

import torch


class TestPredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)

    def predict_and_display(self, text: str, num_tokens: int, top_k: int) -> Tuple[str, str]:
        """Prédit les tokens et formate l'affichage pour Gradio"""
        try:
            # Vérification du modèle
            if not self.model:
                return "Erreur: Modèle non initialisé", "Veuillez d'abord configurer et entraîner le modèle"

            # Prédiction
            predictions, generated_text = self.predict_next_tokens(
                text, 
                num_tokens=int(num_tokens), 
                top_k=int(top_k)
            )

            # Formatage pour l'affichage
            formatted_predictions = self.format_predictions_for_display(predictions)
            return generated_text, formatted_predictions

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return f"Erreur: {str(e)}", "Une erreur est survenue lors de la prédiction"

    def get_top_predictions(
        self, input_ids: torch.Tensor, token_index: int, top_k: int = 5
    ) -> List[Dict]:
        """Get top k predictions for a specific token position"""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits[0, token_index], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                predictions.append({"token": token, "probability": float(prob) * 100})

            return predictions

    def predict_next_tokens(
        self, text: str, num_tokens: int = 5, top_k: int = 5
    ) -> List[Dict]:
        """Predict multiple tokens sequentially"""
        try:
            # Tokenize input text
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = inputs["input_ids"]

            all_predictions = []
            current_text = text

            for _ in range(num_tokens):
                # Get predictions for next token
                predictions = self.get_top_predictions(
                    input_ids, token_index=len(input_ids[0]) - 1, top_k=top_k
                )

                # Add the most likely token
                next_token = predictions[0]["token"]
                current_text += " " + self.tokenizer.convert_tokens_to_string(
                    [next_token]
                )

                # Update input_ids with the predicted token
                next_token_id = self.tokenizer.convert_tokens_to_ids([next_token])[0]
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token_id]])], dim=1
                )

                all_predictions.append(
                    {"position": len(all_predictions) + 1, "predictions": predictions}
                )

            return all_predictions, current_text

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise

    def format_predictions_for_display(self, predictions: List[Dict]) -> str:
        """Format predictions for nice display in Gradio"""
        formatted = ""
        for pred_group in predictions:
            formatted += f"\nPosition {pred_group['position']}:\n"
            for pred in pred_group["predictions"]:
                formatted += f"  {pred['token']}: {pred['probability']:.1f}%\n"
        return formatted
