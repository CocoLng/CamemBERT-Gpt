import logging
import os
import torch
import gradio as gr
from typing import List

from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import wandb

from .data_loader import DataLoader
from .model_config import ModelConfig
from .test_predictor import TestPredictor


class Run_Handler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.model_config = ModelConfig()
        self.training_config = None
        self.test_predictor = None
        self.base_dir = "camembert-training"

    def create_interface(self) -> gr.Blocks:
        """Crée l'interface Gradio complète"""
        with gr.Blocks(title="CamemBERT Training Interface") as interface:
            gr.Markdown("# 🧀 CamemBERT Training Interface")

            with gr.Tab("1. Chargement & Visualisation des Données"):
                with gr.Row():
                    dataset_size = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=1,
                        step=1,
                        label="Taille du Dataset (GB)",
                    )
                    masking_prob = gr.Slider(
                        minimum=0.05,
                        maximum=0.25,
                        value=0.15,
                        step=0.01,
                        label="Probabilité de Masquage (MLM)",
                    )
                    load_btn = gr.Button("Charger Dataset")

                with gr.Row():
                    load_status = gr.Textbox(
                        label="Statut du chargement", interactive=False
                    )

                gr.Markdown("### Test de Masquage")
                with gr.Row():
                    masking_input = gr.Textbox(
                        label="Texte d'entrée (laissez vide pour un texte aléatoire)",
                        placeholder="Entrez un texte en français...",
                        lines=3,
                    )
                text_density = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Densité minimale de texte",
                    info="Ratio minimum de tokens réels vs padding",
                )
                visualize_btn = gr.Button("Visualiser le Masquage")

                with gr.Row():
                    with gr.Column():
                        original_text = gr.Textbox(
                            label="Texte Original", lines=3, interactive=False
                        )
                    with gr.Column():
                        masked_text = gr.Textbox(
                            label="Texte Masqué", lines=3, interactive=False
                        )

            with gr.Tab("2. Configuration du Modèle"):
                with gr.Row():
                    with gr.Column():
                        vocab_size = gr.Slider(
                            minimum=10000,
                            maximum=100000,
                            value=50265,
                            step=1000,
                            label="Taille du Vocabulaire (vocab_size)",
                            info="Défaut RoBERTa: 50265",
                        )
                        hidden_size = gr.Slider(
                            minimum=128,
                            maximum=1024,
                            value=768,
                            step=128,
                            label="Dimension des Embeddings (hidden_size)",
                            info="Défaut RoBERTa: 768",
                        )
                        num_attention_heads = gr.Slider(
                            minimum=4,
                            maximum=16,
                            value=12,
                            step=2,
                            label="Nombre de Têtes d'Attention (num_attention_heads)",
                            info="Défaut RoBERTa: 12",
                        )

                    with gr.Column():
                        num_hidden_layers = gr.Slider(
                            minimum=4,
                            maximum=24,
                            value=12,
                            step=2,
                            label="Nombre de Couches (num_hidden_layers)",
                            info="Défaut RoBERTa: 12",
                        )
                        intermediate_size = gr.Slider(
                            minimum=1024,
                            maximum=4096,
                            value=3072,
                            step=256,
                            label="Taille des Couches Intermédiaires (intermediate_size)",
                            info="Défaut RoBERTa: 3072",
                        )
                        hidden_dropout_prob = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            label="Dropout (hidden_dropout_prob)",
                            info="Défaut RoBERTa: 0.1",
                        )
                        attention_probs_dropout_prob = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            label="Attention Dropout (attention_probs_dropout_prob)",
                            info="Défaut RoBERTa: 0.1",
                        )

                with gr.Row():
                    init_model_btn = gr.Button("Initialiser le Modèle")
                    model_status = gr.Textbox(
                        label="Configuration du Modèle", lines=6, interactive=False
                    )

            with gr.Tab("3. Entraînement"):
                with gr.Row():
                    with gr.Column():
                        output_dir = gr.Textbox(
                            value="camembert-training", 
                            label="Dossier de Sortie"
                        )
                        num_train_epochs = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Nombre d'Epochs",
                        )
                        batch_size = gr.Slider(
                            minimum=8,
                            maximum=64,
                            value=16,
                            step=8,
                            label="Taille des Batchs",
                        )
                        learning_rate = gr.Slider(
                            minimum=1e-6,
                            maximum=1e-4,
                            value=5e-5,
                            step=1e-6,
                            label="Learning Rate",
                        )
                        max_steps = gr.Slider(
                            minimum=1000,
                            maximum=100000,
                            value=10000,
                            step=1000,
                            label="Nombre Maximum de Steps",
                            info="Défaut: 10000"
                        )

                    with gr.Column():
                        use_cuda = gr.Checkbox(
                            value=True,
                            label="Utiliser CUDA (si disponible)",
                            interactive=True,
                        )
                        fp16_training = gr.Checkbox(
                            value=True,
                            label="Utiliser Mixed Precision (FP16)",
                            interactive=True,
                        )
                        weight_decay = gr.Slider(
                            minimum=0.0,
                            maximum=0.1,
                            value=0.01,
                            step=0.01,
                            label="Weight Decay",
                        )
                        warmup_steps = gr.Slider(
                            minimum=0,
                            maximum=20000,
                            value=10000,
                            step=1000,
                            label="Warmup Steps",
                        )
                        gradient_accumulation = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=4,
                            step=1,
                            label="Gradient Accumulation Steps",
                        )
                        num_workers = gr.Slider(
                            minimum=0,
                            maximum=8,
                            value=4,
                            step=1,
                            label="Nombre de Workers pour DataLoader",
                        )

                with gr.Row():
                    wandb_project = gr.Textbox(
                        value="camembert-training", 
                        label="Nom du Projet W&B"
                    )

                # Nouvelle section pour les checkpoints
                gr.Markdown("### Gestion des Checkpoints")
                with gr.Row():
                    with gr.Column():
                        checkpoint_folder = gr.Textbox(
                            label="Dossier des checkpoints",
                            value="camembert-training",
                            interactive=True
                        )
                        available_checkpoints = gr.Dropdown(
                        label="Checkpoints disponibles",
                        choices=["weights"],  # Valeur par défaut
                        interactive=True,
                        allow_custom_value=False  # Empêche les valeurs personnalisées
                    )
                        refresh_checkpoints = gr.Button("Rafraîchir la liste")
                        load_checkpoint_btn = gr.Button("Charger Checkpoint")
                        checkpoint_info = gr.TextArea(
                            label="Informations du Checkpoint",
                            interactive=False,
                            lines=10
                        )

                with gr.Row():
                    start_training_btn = gr.Button("Démarrer l'Entraînement")
                    stop_training_btn = gr.Button("Arrêter l'Entraînement")
                    training_status = gr.Textbox(
                        label="Statut de l'Entraînement", 
                        interactive=False
                    )

            with gr.Tab("4. Test du Modèle"):
                gr.Markdown("### Chargement du Modèle")
                with gr.Row():
                    with gr.Column():
                        model_source = gr.Radio(
                            choices=["Checkpoint", "Weights"],
                            label="Source du modèle",
                            value="Weights"
                        )
                        # Modification du comportement du dossier source
                        available_runs = gr.Dropdown(
                            label="Run disponibles",
                            choices=self._get_run_directories(),
                            interactive=True
                        )
                        test_checkpoints = gr.Dropdown(
                            label="Points de restauration disponibles",
                            choices=[],  # Sera mis à jour dynamiquement
                            interactive=True
                        )
                        refresh_runs = gr.Button("Rafraîchir les runs")
                        refresh_test_checkpoints = gr.Button("Rafraîchir les checkpoints")
                        load_test_model = gr.Button("Charger le modèle")
                        model_load_status = gr.Textbox(
                            label="Statut du chargement",
                            interactive=False
                        )

                gr.Markdown("### Test de Génération de Texte")
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Texte d'entrée",
                            placeholder="Entrez un texte en français...",
                            lines=3,
                        )
                        num_tokens = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Nombre de tokens à prédire",
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Nombre de prédictions par position",
                        )
                        predict_btn = gr.Button("Prédire")

                with gr.Row():
                    with gr.Column():
                        predicted_text = gr.Textbox(
                            label="Texte Généré", lines=3, interactive=False
                        )
                    with gr.Column():
                        predictions_display = gr.Textbox(
                            label="Détails des Prédictions", lines=10, interactive=False
                        )

            # Event Handlers
            load_btn.click(
                fn=self._load_dataset_with_vocab_check,
                inputs=[dataset_size, masking_prob, vocab_size], 
                outputs=[load_status],
            )

            visualize_btn.click(
                fn=self.data_loader.visualize_with_density,
                inputs=[masking_input, text_density],
                outputs=[original_text, masked_text],
            )

            init_model_btn.click(
                fn=self._initialize_model_with_vocab,
                inputs=[
                    vocab_size,
                    hidden_size,
                    num_attention_heads,
                    num_hidden_layers,
                    intermediate_size,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                ],
                outputs=[model_status],
            )

            # Event handlers pour les checkpoints
            refresh_checkpoints.click(
                fn=self._get_available_checkpoints,
                inputs=[checkpoint_folder],
                outputs=[available_checkpoints],
            )
            
            # Event Handlers pour la nouvelle interface
            refresh_runs.click(
                fn=lambda: self._get_run_directories(),
                outputs=[available_runs]
            )
            
            # Mise à jour automatique des checkpoints quand on change de run
            available_runs.change(
                fn=self._get_available_checkpoints,
                inputs=[available_runs],
                outputs=[test_checkpoints]
            )

            refresh_test_checkpoints.click(
                fn=self._get_available_checkpoints,
                inputs=[available_runs],
                outputs=[test_checkpoints]
            )

            # Modification du handler de chargement
            load_test_model.click(
                fn=self._load_model_for_testing,
                inputs=[model_source, available_runs, test_checkpoints],
                outputs=[model_load_status]
            )
            
            load_checkpoint_btn.click(
                fn=self._get_checkpoint_info,
                inputs=[checkpoint_folder, available_checkpoints],
                outputs=[checkpoint_info]
            )
            

            predict_btn.click(
                fn=lambda *args: self.test_predictor.predict_and_display(*args)
                if self.test_predictor
                else ("Modèle non initialisé", "Veuillez d'abord charger un modèle"),
                inputs=[input_text, num_tokens, top_k],
                outputs=[predicted_text, predictions_display],
            )

            start_training_btn.click(
                fn=lambda *args: self.training_config.start_training(*args)
                if self.training_config
                else "❌ Configuration non initialisée",
                inputs=[
                    output_dir,
                    num_train_epochs,
                    batch_size,
                    learning_rate,
                    weight_decay,
                    warmup_steps,
                    gradient_accumulation,
                    wandb_project,
                    use_cuda,
                    fp16_training,
                    num_workers,
                    max_steps,
                ],
                outputs=[training_status],
            )

            stop_training_btn.click(
                fn=lambda: self.training_config.stop_training()
                if self.training_config
                else "❌ Configuration non initialisée",
                outputs=[training_status],
            )

            return interface

    def _get_run_directories(self) -> List[str]:
        """Récupère tous les dossiers de runs valides"""
        try:
            if not os.path.exists(self.base_dir):
                self.logger.warning(f"Dossier de base non trouvé: {self.base_dir}")
                return []
                
            run_dirs = []
            # Liste tous les dossiers cam_runX
            for item in os.listdir(self.base_dir):
                if item.startswith("cam_run"):
                    full_path = os.path.join(self.base_dir, item)
                    if os.path.isdir(full_path):
                        run_dirs.append(full_path)
                        
            return sorted(run_dirs, key=lambda x: int(x.split("cam_run")[-1]))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture des runs: {e}")
            return []

    def _get_available_checkpoints(self, folder: str) -> List[str]:
        """Récupère la liste des checkpoints disponibles pour un dossier donné"""
        try:
            checkpoints = []
            
            # 1. Vérifier si c'est un chemin direct vers un run
            if os.path.basename(folder).startswith("cam_run"):
                base_path = folder
            else:
                # 2. Sinon, construire le chemin complet
                if not folder.startswith(self.base_dir):
                    base_path = os.path.join(self.base_dir, folder)
                else:
                    base_path = folder

            # 3. Vérifier si le dossier existe
            if not os.path.exists(base_path):
                self.logger.warning(f"Dossier non trouvé: {base_path}")
                return []

            # 4. Ajouter le dossier weights s'il existe et contient les fichiers nécessaires
            weights_path = os.path.join(base_path, "weights")
            if os.path.exists(weights_path) and self._is_valid_model_directory(weights_path):
                checkpoints.append(f"weights")

            # 5. Ajouter les checkpoints valides
            for item in os.listdir(base_path):
                if item.startswith("checkpoint-"):
                    checkpoint_path = os.path.join(base_path, item)
                    if os.path.isdir(checkpoint_path) and self._is_valid_model_directory(checkpoint_path):
                        checkpoints.append(item)

            # 6. Trier les checkpoints
            sorted_checkpoints = sorted(checkpoints, 
                                     key=lambda x: int(x.split('-')[1]) if x != "weights" else float('inf'))
            
            self.logger.info(f"Checkpoints trouvés dans {base_path}: {sorted_checkpoints}")
            return sorted_checkpoints

        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture des checkpoints: {e}")
            return []
    
    def _get_checkpoint_info(self, folder: str, checkpoint: str) -> str:
        """Récupère les informations d'un checkpoint"""
        try:
            checkpoint_path = os.path.join(folder, checkpoint)
            
            # Lecture du rapport de métriques
            metrics_path = os.path.join(checkpoint_path, "metrics_report.txt")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = f.read()
            else:
                metrics = "Rapport de métriques non disponible"
                
            # Lecture de l'état de l'entraînement
            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
            if os.path.exists(trainer_state_path):
                state = torch.load(trainer_state_path)
                training_info = (
                    f"État de l'entraînement:\n"
                    f"- Step: {state.get('global_step', 'N/A')}\n"
                    f"- Epoch: {state.get('epoch', 'N/A')}\n"
                )
            else:
                training_info = "État de l'entraînement non disponible"
                
            return f"{training_info}\n\nMétriques détaillées:\n{metrics}"
        except Exception as e:
            return f"Erreur lors de la lecture des informations: {str(e)}"
        
    def _load_model_for_testing(self, source: str, folder: str, checkpoint: str) -> str:
        """Charge le modèle pour le test depuis un checkpoint ou les weights"""
        try:
            if not folder or not checkpoint:
                return "❌ Veuillez sélectionner un dossier et un point de restauration"

            # Déterminer le chemin selon la source
            if source == "Weights":
                path = os.path.join(folder, "weights")
            else:
                path = os.path.join(folder, checkpoint)

            if not os.path.exists(path):
                return f"❌ Chemin non trouvé: {path}"

            # Charger le modèle et le tokenizer
            model = RobertaForMaskedLM.from_pretrained(path)
            tokenizer = RobertaTokenizerFast.from_pretrained(path)
            
            # Déplacer le modèle sur GPU si disponible
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Mettre à jour le test_predictor
            self.test_predictor = TestPredictor(model, tokenizer)
            
            self.logger.info(f"Modèle chargé depuis: {path}")
            return "✅ Modèle chargé avec succès"
            
        except Exception as e:
            error_msg = f"❌ Erreur lors du chargement: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        
    def _is_valid_model_directory(self, directory: str) -> bool:
        """Vérifie si un dossier contient un modèle valide"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        return all(os.path.exists(os.path.join(directory, f)) for f in required_files)

    def _load_model_and_tokenizer(self, path: str) -> None:
        """Charge le modèle et le tokenizer depuis un checkpoint ou weights"""
        try:
            # Utilisation de from_pretrained pour charger le modèle
            model = RobertaForMaskedLM.from_pretrained(path)
            tokenizer = RobertaTokenizerFast.from_pretrained(path)
            
            # Mise à jour du test_predictor
            self.test_predictor = TestPredictor(model, tokenizer)
            
            self.logger.info(f"Modèle chargé depuis: {path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

    # Pour fixer les erreurs CUDA
    def _load_dataset_with_vocab_check(self, size: float, prob: float, vocab_size: int) -> str:
        """Load dataset with vocabulary size synchronization"""
        try:
            # Update current vocab size
            self._current_vocab_size = vocab_size
            
            # Initialize data loader with correct vocab size
            status = self.data_loader.load_with_masking(
                size=size,
                prob=prob,
                vocab_size=vocab_size
            )
            
            self.logger.info(f"Dataset loaded with vocab size: {vocab_size}")
            return status
            
        except Exception as e:
            error_msg = f"Error loading dataset: {e}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    def _initialize_model_with_vocab(
        self,
        vocab_size: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ) -> str:
        """Initialize model with vocabulary synchronization"""
        try:
            # Check if vocab size changed
            if vocab_size != self._current_vocab_size:
                # Reload dataset with new vocab size if needed
                if self.data_loader.is_ready():
                    self.logger.info("Reloading dataset with new vocab size")
                    self.data_loader.setup_for_training(vocab_size)
                self._current_vocab_size = vocab_size

            return self.model_config.initialize_full_config(
                vocab_size,
                hidden_size,
                num_attention_heads,
                num_hidden_layers,
                intermediate_size,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                run_handler=self
            )

        except Exception as e:
            error_msg = f"Error initializing model: {e}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    def _verify_vocab_consistency(self) -> bool:
        """Verify that vocabulary sizes are consistent across components"""
        if not self.data_loader.is_ready() or not self.model_config.config:
            return False

        vocab_matches = (
            self.data_loader.vocab_size == self.model_config.config.vocab_size == 
            self._current_vocab_size
        )
        
        if not vocab_matches:
            self.logger.error(
                f"Vocabulary size mismatch: "
                f"DataLoader={self.data_loader.vocab_size}, "
                f"Model={self.model_config.config.vocab_size}, "
                f"Current={self._current_vocab_size}"
            )
            
        return vocab_matches

    def run(self) -> None:
        """Lance l'interface Gradio"""
        try:
            interface = self.create_interface()
            interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
        except KeyboardInterrupt:
            self.logger.info("Interface arrêtée par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur lors du lancement de l'interface: {e}")
