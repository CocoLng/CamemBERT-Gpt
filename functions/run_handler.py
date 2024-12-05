import logging

import gradio as gr

from .data_loader import DataLoader
from .model_config import ModelConfig


class Run_Handler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.model_config = ModelConfig()
        self.training_config = None
        self.test_predictor = None

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

            # Tab 2: Model Configuration
            with gr.Tab("2. Configuration du Modèle"):
                with gr.Row():
                    with gr.Column():
                        # Architecture parametres
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

            # Tab 3: Training Configuration and Monitoring
            with gr.Tab("3. Entraînement"):
                with gr.Row():
                    with gr.Column():
                        output_dir = gr.Textbox(
                            value="camembert-training", label="Dossier de Sortie"
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
                        value="camembert-training", label="Nom du Projet W&B"
                    )

                with gr.Row():
                    start_training_btn = gr.Button("Démarrer l'Entraînement")
                    stop_training_btn = gr.Button("Arrêter l'Entraînement")
                    training_status = gr.Textbox(
                        label="Statut de l'Entraînement", interactive=False
                    )

            # Tab 4: Model Testing
            with gr.Tab("4. Test du Modèle"):
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
                fn=self.data_loader.load_with_masking,
                inputs=[dataset_size, masking_prob],
                outputs=[load_status],
            )

            visualize_btn.click(
                fn=self.data_loader.visualize_with_density,
                inputs=[masking_input, text_density],
                outputs=[original_text, masked_text],
            )

            init_model_btn.click(
                fn=lambda *args: self.model_config.initialize_full_config(
                    *args, run_handler=self
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
                outputs=[model_status],
            )

            predict_btn.click(
                fn=lambda *args: self.test_predictor.predict_and_display(*args)
                if self.test_predictor
                else "Model not initialized",
                inputs=[input_text, num_tokens, top_k],
                outputs=[predicted_text, predictions_display],
            )

            # Training Events
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
                ],
                outputs=[training_status],
            )

            stop_training_btn.click(
                fn=lambda: self.training_config.stop_training()
                if self.training_config
                else "❌ Configuration non initialisée",
                outputs=[training_status],
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
                    max_steps,  # Ajout du nouveau paramètre
                ],
                outputs=[training_status],
            )

            return interface

    def run(self) -> None:
        """Lance l'interface Gradio"""
        try:
            interface = self.create_interface()
            interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
        except KeyboardInterrupt:
            self.logger.info("Interface arrêtée par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur lors du lancement de l'interface: {e}")
