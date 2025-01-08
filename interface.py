import logging
import os
from typing import List

import gradio as gr
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from functions.data_loader import DataLoader, DatasetConfig
from functions.fine_tuning import Finetune_NLI
from functions.fine_tuning_saver import FineTuningSaver  
from functions.masking_monitor import MaskingHandler 
from functions.model_config import ModelConfig
from functions.test_predictor import TestPredictor

# Appélé depuis le fichier main.py

class Run_Handler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader(dataset_config=DatasetConfig())
        self.model_config = ModelConfig()
        self.training_config = None
        self.test_predictor = None
        self.masking_handler = MaskingHandler(self.data_loader)
        self.base_dir = "camembert-training"
        self.nli_dir = "camembert-nli"
        
        # Create necessary directories
        os.makedirs(self.nli_dir, exist_ok=True)
        
        # Initialize both fine_tuning and fine_tuning_saver
        self.fine_tuning = Finetune_NLI(base_dir=self.nli_dir)  
        self.fine_tuning_saver = FineTuningSaver(run_dir=self.nli_dir)

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(title="CamemBERT Training Interface") as interface:
            gr.Markdown("# 🧀 CamemBERT Training Interface")

            with gr.Tab("1. Chargement & Visualisation des Données"):
                with gr.Row():
                    dataset_choice = gr.Dropdown(
                        choices=["mOSCAR (default)", "OSCAR-2301"],
                        value="mOSCAR (default)",
                        label="Source des Données",
                    )

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

                with gr.Row():
                    load_btn = gr.Button("Charger Dataset")
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
                            minimum=32000,
                            maximum=50265,
                            value=50265,
                            step=1000,
                            label="Taille du Vocabulaire (vocab_size)",
                            info="Choix article: 50265",
                        )
                        hidden_size = gr.Slider(
                            minimum=128,
                            maximum=1024,
                            value=768,
                            step=128,
                            label="Dimension des Embeddings (hidden_size)",
                            info="Choix article: 768",
                        )
                        num_attention_heads = gr.Slider(
                            minimum=4,
                            maximum=16,
                            value=12,
                            step=2,
                            label="Nombre de Têtes d'Attention (num_attention_heads)",
                            info="Choix article: 12",
                        )

                    with gr.Column():
                        num_hidden_layers = gr.Slider(
                            minimum=4,
                            maximum=24,
                            value=12,
                            step=2,
                            label="Nombre de Couches (num_hidden_layers)",
                            info="Choix article: 12",
                        )
                        intermediate_size = gr.Slider(
                            minimum=1024,
                            maximum=4096,
                            value=3072,
                            step=256,
                            label="Taille des Couches Intermédiaires (intermediate_size)",
                            info="Choix article: 3072",
                        )
                        hidden_dropout_prob = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            label="Dropout (hidden_dropout_prob)",
                            info="Choix article: 0.1",
                        )
                        attention_probs_dropout_prob = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            label="Attention Dropout (attention_probs_dropout_prob)",
                            info="Choix article: 0.1",
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
                            value="camembert-training", label="Dossier de Sortie"
                        )
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

                    with gr.Column():
                        training_config_display = gr.TextArea(
                            label="Configuration d'Entraînement Calculée",
                            interactive=False,
                            lines=12,
                        )
                        wandb_project = gr.Textbox(
                            value="camembert-training", label="Nom du Projet W&B"
                        )

                gr.Markdown("### Gestion des Checkpoints")
                with gr.Row():
                    with gr.Column():
                        checkpoint_folder = gr.Textbox(
                            label="Dossier des checkpoints",
                            value="camembert-training",
                            interactive=True,
                        )
                        available_checkpoints = gr.Dropdown(
                            label="Checkpoints disponibles",
                            choices=["weights"], 
                            interactive=True,
                            allow_custom_value=False,  
                        )
                        refresh_checkpoints = gr.Button("Rafraîchir la liste")
                        load_checkpoint_btn = gr.Button("Charger Checkpoint")
                        checkpoint_info = gr.TextArea(
                            label="Informations du Checkpoint",
                            interactive=False,
                            lines=8,
                        )

                with gr.Row():
                    start_training_btn = gr.Button("Démarrer l'Entraînement")
                    stop_training_btn = gr.Button("Arrêter l'Entraînement")
                    training_status = gr.Textbox(
                        label="Statut de l'Entraînement", interactive=False
                    )

            with gr.Tab("4. Test du Modèle"):
                gr.Markdown("### Chargement du Modèle")
                with gr.Row():
                    with gr.Column():
                        model_source = gr.Radio(
                            choices=["Checkpoint", "Weights"],
                            label="Source du modèle",
                            value="Weights",
                        )
                        available_runs = gr.Dropdown(
                            label="Run disponibles",
                            choices=self._get_run_directories(),
                            interactive=True,
                            value=None, 
                        )
                        test_checkpoints = gr.Dropdown(
                            label="Points de restauration disponibles",
                            choices=[],
                            interactive=True,
                            visible=True,
                        )
                        refresh_runs = gr.Button("Rafraîchir les runs")
                        load_test_model = gr.Button("Charger le modèle")
                        model_load_status = gr.Textbox(
                            label="Statut du chargement", interactive=False
                        )

                # Event handlers
                model_source.change(
                    fn=lambda source: gr.update(visible=source == "Checkpoint"),
                    inputs=[model_source],
                    outputs=[test_checkpoints],
                )

                def update_checkpoints(run_name):
                    if isinstance(run_name, list):
                        run_name = run_name[0] if run_name else None
                    return gr.Dropdown(
                        choices=self._get_available_checkpoints(str(run_name))
                    )

                def handle_model_loading(source, run_name, checkpoint):
                    if isinstance(run_name, list):
                        run_name = run_name[0] if run_name else None

                    if not run_name:
                        return "❌ Veuillez sélectionner un run"

                    run_dir = os.path.join(self.base_dir, str(run_name))

                    if source == "Checkpoint":
                        if not checkpoint:
                            return "❌ Veuillez sélectionner un checkpoint"
                        path = os.path.join(run_dir, str(checkpoint))
                    else:
                        path = os.path.join(run_dir, "weights")

                    return self._load_model_for_testing(source, run_dir, path)

                available_runs.change(
                    fn=update_checkpoints,
                    inputs=[available_runs],
                    outputs=[test_checkpoints],
                )

                refresh_runs.click(
                    fn=lambda: gr.Dropdown(choices=self._get_run_directories()),
                    outputs=[available_runs],
                )

                load_test_model.click(
                    fn=handle_model_loading,
                    inputs=[model_source, available_runs, test_checkpoints],
                    outputs=[model_load_status],
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

            with gr.Tab("5. Fine-Tuning NLI"):
                gr.Markdown("### Configuration du Modèle Pré-entraîné")
                with gr.Row():
                    with gr.Column():
                        model_repo = gr.Dropdown(
                            label="Dossier du modèle",
                            choices=self._get_run_directories(),  # Réutilise la fonction de la partie 4
                            interactive=True,
                            value=None,
                            info="Sélectionner le dossier du modèle dans camembert-training/ (ex: cam_run24)"
                        )
                        weights_path = gr.Textbox(
                            label="Chemin des poids",
                            value="weights/model.safetensors",
                            info="Chemin relatif depuis le dossier du modèle"
                        )
                    
                    with gr.Column():
                        config_path = gr.Textbox(
                            label="Chemin de la configuration",
                            value="weights/config.json",
                            info="Chemin relatif depuis le dossier du modèle"
                        )
                        tokenizer_name = gr.Textbox(
                            value="camembert-base",
                            label="Nom du tokenizer"
                        )
                        load_model_btn = gr.Button("Charger le Modèle")
                        model_status = gr.Textbox(
                            label="Statut du Modèle",
                            interactive=False
                        )

                    # Rafraîchir la liste des runs
                    refresh_runs = gr.Button("🔄 Rafraîchir")
                    refresh_runs.click(
                        fn=lambda: gr.Dropdown(choices=self._get_run_directories()),
                        outputs=[model_repo]
                    )

                    # Chargement du modèle
                    load_model_btn.click(
                        fn=self.fine_tuning.initialize_model,
                        inputs=[
                            model_repo,
                            weights_path,
                            config_path,
                            tokenizer_name
                        ],
                        outputs=[model_status]
                    )

                gr.Markdown("### Configuration du Dataset NLI")
                with gr.Row():
                    with gr.Column():
                        dataset_name = gr.Textbox(
                            value="facebook/xnli",
                            label="Nom du Dataset"
                        )
                        language = gr.Dropdown(
                            choices=["fr", "en", "es"],
                            value="fr",
                            label="Langue"
                        )
                        test_split = gr.Slider(
                            minimum=0.1,
                            maximum=0.4,
                            value=0.2,
                            step=0.05,
                            label="Proportion Test Set"
                        )
                    
                    with gr.Column():
                        prepare_data_btn = gr.Button("Préparer les Données")
                        data_status = gr.Textbox(
                            label="Statut des Données",
                            interactive=False
                        )
                        dataset_info = gr.DataFrame(
                            label="Aperçu du Dataset",
                            interactive=False
                        )

                gr.Markdown("### Configuration de l'Entraînement")
                with gr.Row():
                    with gr.Column():
                        num_epochs = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Nombre d'Epochs"
                        )
                        batch_size = gr.Slider(
                            minimum=4,
                            maximum=32,
                            value=8,
                            step=4,
                            label="Taille des Batchs"
                        )
                        learning_rate = gr.Slider(
                            minimum=1e-6,
                            maximum=1e-4,
                            value=1e-5,
                            step=1e-6,
                            label="Learning Rate"
                        )
                        num_labels = gr.Slider(
                            minimum=2,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Nombre de Classes"
                        )

                    with gr.Column():
                        save_dir = gr.Textbox(
                            value="camembert-nli",
                            label="Dossier de Sauvegarde"
                        )
                        use_tensorboard = gr.Checkbox(
                            value=True,
                            label="Utiliser TensorBoard"
                        )

                with gr.Row():
                    start_ft_btn = gr.Button("Démarrer le Fine-tuning")
                    stop_ft_btn = gr.Button("Arrêter le Fine-tuning")
                    training_status = gr.Textbox(
                        label="Statut de l'Entraînement",
                        interactive=False
                    )

                gr.Markdown("### Évaluation et Métriques")
                with gr.Row():
                    with gr.Column():
                        evaluate_btn = gr.Button("Évaluer le Modèle")
                        metrics_display = gr.DataFrame(
                            label="Métriques d'Évaluation",
                            interactive=False
                        )
                    with gr.Column():
                        confusion_matrix = gr.Plot(
                            label="Matrice de Confusion"
                        )
                        accuracy_plot = gr.Plot(
                            label="Courbe d'Apprentissage"
                        )

                # Event Handlers
                load_model_btn.click(
                    fn=self.fine_tuning.initialize_model,
                    inputs=[model_repo, weights_path, config_path, tokenizer_name],
                    outputs=[model_status]
                )

                prepare_data_btn.click(
                    fn=self.fine_tuning.prepare_dataset,
                    inputs=[dataset_name, language, test_split],
                    outputs=[data_status, dataset_info]
                )

                start_ft_btn.click(
                    fn=self.fine_tuning.start_training,
                    inputs=[
                        num_epochs, batch_size, learning_rate, num_labels,
                        save_dir, use_tensorboard
                    ],
                    outputs=[training_status]
                )

                stop_ft_btn.click(
                    fn=self.fine_tuning.stop_training,
                    outputs=[training_status]
                )

                evaluate_btn.click(
                    fn=self.fine_tuning.evaluate_model,
                    outputs=[metrics_display, confusion_matrix, accuracy_plot]
                )

            # Event Handlers
            load_btn.click(
                fn=self.masking_handler.initialize_and_load_dataset,
                inputs=[dataset_choice, dataset_size, masking_prob],
                outputs=[load_status],
            )

            visualize_btn.click(
                fn=self.masking_handler.visualize_with_density,
                inputs=[masking_input, text_density],
                outputs=[original_text, masked_text],
            )

            init_model_btn.click(
                fn=lambda vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                inter_size,
                hidden_dropout,
                attn_dropout: self.model_config.initialize_full_config(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    num_hidden_layers=num_layers,
                    intermediate_size=inter_size,
                    hidden_dropout_prob=hidden_dropout,
                    attention_probs_dropout_prob=attn_dropout,
                    run_handler=self,
                ),
                inputs=[
                    vocab_size,
                    hidden_size,
                    num_attention_heads,
                    num_hidden_layers,
                    intermediate_size,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                ],
                outputs=[model_status, training_config_display],
            )
            # Event handlers pour les checkpoints
            refresh_checkpoints.click(
                fn=self._get_available_checkpoints,
                inputs=[checkpoint_folder],
                outputs=[available_checkpoints],
            )

            load_checkpoint_btn.click(
                fn=self._get_checkpoint_info,
                inputs=[checkpoint_folder, available_checkpoints],
                outputs=[checkpoint_info],
            )

            start_training_btn.click(
                fn=lambda *args: self.training_config.start_training(*args)
                if self.training_config
                else "❌ Configuration non initialisée",
                inputs=[output_dir, wandb_project, use_cuda, fp16_training],
                outputs=[training_status],
            )

            stop_training_btn.click(
                fn=lambda: self.training_config.stop_training()
                if self.training_config
                else "❌ Configuration non initialisée",
                outputs=[training_status],
            )

            predict_btn.click(
                fn=lambda *args: self.test_predictor.predict_and_display(*args)
                if self.test_predictor
                else ("Modèle non initialisé", "Veuillez d'abord charger un modèle"),
                inputs=[input_text, num_tokens, top_k],
                outputs=[predicted_text, predictions_display],
            )
            
            return interface

    def _get_run_directories(self) -> List[str]:
        """Récupère la liste des runs disponibles"""
        try:
            runs = []
            if os.path.exists(self.base_dir):
                for item in os.listdir(self.base_dir):
                    if item.startswith("cam_run"):
                        if os.path.isdir(os.path.join(self.base_dir, item)):
                            runs.append(item)
            return sorted(runs, key=lambda x: int(x.split("cam_run")[-1]))
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture des runs: {e}")
            return []

    def _get_available_checkpoints(self, run_name: str) -> List[str]:
        """Récupère les checkpoints disponibles pour un run"""
        try:
            if not run_name:
                return []

            checkpoints = []
            run_dir = os.path.join(self.base_dir, str(run_name))

            # Vérifier le dossier weights
            weights_path = os.path.join(run_dir, "weights")
            if os.path.exists(weights_path):
                checkpoints.append("weights")

            # Chercher les checkpoints
            for item in os.listdir(run_dir):
                if item.startswith("checkpoint-"):
                    checkpoint_path = os.path.join(run_dir, item)
                    if os.path.isdir(checkpoint_path):
                        checkpoints.append(item)

            return sorted(checkpoints)
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
                with open(metrics_path, "r") as f:
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

    def _load_model_for_testing(self, source: str, folder: str, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"❌ Chemin non trouvé: {path}"
                
            # Charger le tokenizer
            from transformers import CamembertTokenizerFast
            tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
            
            # Charger le modèle
            model = RobertaForMaskedLM.from_pretrained(path)
            
            if torch.cuda.is_available():
                model = model.cuda()

            self.test_predictor = TestPredictor(model, tokenizer)
            return "✅ Modèle chargé avec succès"

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {str(e)}")
            return f"❌ Erreur: {str(e)}"

    def _is_valid_model_directory(self, directory: str) -> bool:
        """Vérifie si un dossier contient un modèle valide"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
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

    def run(self) -> None:
        """Lance l'interface Gradio"""
        try:
            interface = self.create_interface()
            interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
        except KeyboardInterrupt:
            self.logger.info("Interface arrêtée par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur lors du lancement de l'interface: {e}")