import gradio as gr
import logging
from typing import Dict, Tuple
from .data_loader import DataLoader
from .model_config import ModelConfig, ModelArguments
from .training_config import TrainingConfig, TrainerArguments
import wandb
import os

class Run_Handler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.model_config = ModelConfig()
        self.training_config = None  # Will be initialized after model config
        
    def create_interface(self) -> gr.Blocks:
        """Crée l'interface Gradio complète"""
        with gr.Blocks(title="CamemBERT Training Interface") as interface:
            gr.Markdown("# 🧀 CamemBERT Training Interface")
            
            # Tab 1: Data Loading (existing code)
            with gr.Tab("1. Chargement & Visualisation des Données"):
                with gr.Row():
                    dataset_size = gr.Slider(
                        minimum=1, maximum=100, value=1, step=1,
                        label="Taille du Dataset (GB)"
                    )
                    load_btn = gr.Button("Charger Dataset")
                
                with gr.Row():
                    load_status = gr.Textbox(
                        label="Statut du chargement",
                        interactive=False
                    )
                
                gr.Markdown("### Test de Masquage")
                with gr.Row():
                    input_text = gr.Textbox(
                        label="Texte d'entrée (laissez vide pour un texte aléatoire)",
                        placeholder="Entrez un texte en français...",
                        lines=3
                    )
                    visualize_btn = gr.Button("Visualiser le Masquage")
                
                with gr.Row():
                    with gr.Column():
                        original_text = gr.Textbox(
                            label="Texte Original",
                            lines=3,
                            interactive=False
                        )
                    with gr.Column():
                        masked_text = gr.Textbox(
                            label="Texte Masqué",
                            lines=3,
                            interactive=False
                        )
            
            # Tab 2: Model Configuration
            with gr.Tab("2. Configuration du Modèle"):
                with gr.Row():
                    with gr.Column():
                        vocab_size = gr.Number(
                            value=50265,
                            label="Taille du Vocabulaire",
                            precision=0
                        )
                        hidden_size = gr.Number(
                            value=768,
                            label="Dimension des Embeddings",
                            precision=0
                        )
                        num_attention_heads = gr.Number(
                            value=12,
                            label="Nombre de Têtes d'Attention",
                            precision=0
                        )
                        num_hidden_layers = gr.Number(
                            value=12,
                            label="Nombre de Couches",
                            precision=0
                        )
                    
                    with gr.Column():
                        intermediate_size = gr.Number(
                            value=3072,
                            label="Taille Intermédiaire",
                            precision=0
                        )
                        hidden_dropout_prob = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            label="Dropout"
                        )
                        attention_probs_dropout_prob = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            label="Attention Dropout"
                        )
                
                with gr.Row():
                    init_model_btn = gr.Button("Initialiser le Modèle")
                    model_status = gr.Textbox(
                        label="Statut de la Configuration",
                        interactive=False
                    )
            
            # Tab 3: Training Configuration and Monitoring
            with gr.Tab("3. Entraînement"):
                with gr.Row():
                    with gr.Column():
                        output_dir = gr.Textbox(
                            value="camembert-fr",
                            label="Dossier de Sortie"
                        )
                        num_train_epochs = gr.Number(
                            value=3,
                            label="Nombre d'Epochs",
                            precision=0
                        )
                        batch_size = gr.Number(
                            value=16,
                            label="Taille des Batchs",
                            precision=0
                        )
                        learning_rate = gr.Number(
                            value=5e-5,
                            label="Learning Rate",
                        )
                    
                    with gr.Column():
                        weight_decay = gr.Number(
                            value=0.01,
                            label="Weight Decay"
                        )
                        warmup_steps = gr.Number(
                            value=10000,
                            label="Warmup Steps",
                            precision=0
                        )
                        gradient_accumulation = gr.Number(
                            value=4,
                            label="Gradient Accumulation Steps",
                            precision=0
                        )
                        wandb_project = gr.Textbox(
                            value="camembert-fr",
                            label="Nom du Projet W&B"
                        )
                
                with gr.Row():
                    start_training_btn = gr.Button("Démarrer l'Entraînement")
                    stop_training_btn = gr.Button("Arrêter l'Entraînement")
                    training_status = gr.Textbox(
                        label="Statut de l'Entraînement",
                        interactive=False
                    )
                
                with gr.Row():
                    # Progress monitoring
                    training_progress = gr.Plot(label="Courbe d'Apprentissage")
                    current_metrics = gr.JSON(label="Métriques Actuelles")
            
            # Events for Tab 1 (existing)
            load_btn.click(
                fn=self.data_loader.load_streaming_dataset,
                inputs=[dataset_size],
                outputs=[load_status]
            )
            
            visualize_btn.click(
                fn=self.data_loader.visualize_masking,
                inputs=[input_text],
                outputs=[original_text, masked_text]
            )
            
            # Events for Tab 2
            def initialize_model_config(vocab_size, hidden_size, num_attention_heads, 
                                     num_hidden_layers, intermediate_size, hidden_dropout_prob,
                                     attention_probs_dropout_prob):
                try:
                    self.model_config.initialize_config(
                        vocab_size=int(vocab_size),
                        hidden_size=int(hidden_size),
                        num_attention_heads=int(num_attention_heads),
                        num_hidden_layers=int(num_hidden_layers),
                        intermediate_size=int(intermediate_size),
                        hidden_dropout_prob=hidden_dropout_prob,
                        attention_probs_dropout_prob=attention_probs_dropout_prob
                    )
                    
                    # Initialize training config after model config
                    self.training_config = TrainingConfig(self.model_config, self.data_loader)
                    
                    return "✅ Configuration du modèle initialisée avec succès!"
                except Exception as e:
                    return f"❌ Erreur lors de l'initialisation: {str(e)}"
            
            init_model_btn.click(
                fn=initialize_model_config,
                inputs=[vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
                       intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob],
                outputs=[model_status]
            )
            
            # Events for Tab 3
            def start_training(output_dir, num_train_epochs, batch_size, learning_rate,
                             weight_decay, warmup_steps, gradient_accumulation, wandb_project):
                try:
                    if not self.training_config:
                        return "❌ Veuillez d'abord initialiser la configuration du modèle"
                    
                    # Initialize wandb
                    wandb.init(project=wandb_project, name=f"training-run-{output_dir}")
                    
                    # Setup training arguments
                    self.training_config.setup_training_arguments(
                        output_dir=output_dir,
                        num_train_epochs=int(num_train_epochs),
                        per_device_train_batch_size=int(batch_size),
                        learning_rate=float(learning_rate),
                        weight_decay=float(weight_decay),
                        warmup_steps=int(warmup_steps),
                        gradient_accumulation_steps=int(gradient_accumulation)
                    )
                    
                    # Setup trainer
                    self.training_config.setup_trainer()
                    
                    # Start training in a separate thread
                    import threading
                    self.training_thread = threading.Thread(target=self.training_config.train)
                    self.training_thread.start()
                    
                    return "✅ Entraînement démarré!"
                except Exception as e:
                    return f"❌ Erreur lors du démarrage: {str(e)}"
            
            def stop_training():
                try:
                    if hasattr(self, 'training_thread') and self.training_thread.is_alive():
                        # Implement graceful shutdown
                        wandb.finish()
                        return "✅ Arrêt de l'entraînement demandé"
                    return "❌ Aucun entraînement en cours"
                except Exception as e:
                    return f"❌ Erreur lors de l'arrêt: {str(e)}"
            
            start_training_btn.click(
                fn=start_training,
                inputs=[output_dir, num_train_epochs, batch_size, learning_rate,
                       weight_decay, warmup_steps, gradient_accumulation, wandb_project],
                outputs=[training_status]
            )
            
            stop_training_btn.click(
                fn=stop_training,
                outputs=[training_status]
            )
            
            return interface

    def run(self) -> None:
        """Lance l'interface Gradio"""
        try:
            interface = self.create_interface()
            interface.launch(
                share=False,
                server_name="0.0.0.0",
                server_port=7860
            )
        except KeyboardInterrupt:
            self.logger.info("Interface arrêtée par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur lors du lancement de l'interface: {e}")