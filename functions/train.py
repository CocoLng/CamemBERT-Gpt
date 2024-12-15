import logging
import os
import sys

import torch
import wandb
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments

from .masking_monitor import (
    MaskingHandler,
    MaskingMonitorCallback,
)
from .training_saver import TrainingSaver  


class CustomTrainer(TrainingSaver, Trainer):
    def __init__(self, data_loader=None, checkpoint_steps=4000, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Vérification des arguments requis pour l'entraînement
        if "args" not in kwargs:
            raise ValueError("Training arguments (args) must be provided")

        # Configuration des attributs de base
        self.checkpoint_steps = checkpoint_steps
        self.tokens_processed = 0
        self._initial_log_done = False
        self.masking_handler = kwargs.pop("masking_handler", None)

        # Configuration du data_loader et ses attributs
        self.data_loader = data_loader
        self.dataset_size = None
        self.processing_class = None

        if data_loader:
            self.dataset_size = getattr(data_loader, "_dataset_size", 0)
            self.processing_class = getattr(data_loader, "tokenizer", None)
            if "train_dataset" not in kwargs or kwargs["train_dataset"] is None:
                kwargs["train_dataset"] = data_loader.dataset

        # Initialisation des classes parentes
        TrainingSaver.__init__(
            self,
            run_dir=kwargs["args"].output_dir,
            dataset_size=self.dataset_size,
            processing_class=self.processing_class,
        )
        Trainer.__init__(self, **kwargs)

        # Sauvegarde des informations initiales du modèle
        if hasattr(self, "model") and self.model is not None:
            self._save_model_info(self.run_dir)

    def training_step(self, model, inputs, return_loss=True):
        """Étape d'entraînement avec surveillance et performance optimales."""
        try:
            required_keys = {"input_ids", "attention_mask", "labels"}
            if not all(k in inputs for k in required_keys):
                raise ValueError(
                    f"Missing required keys: {required_keys - set(inputs.keys())}"
                )

            # Calcul efficace des tokens traités
            tokens_in_batch = inputs["input_ids"].size(0) * inputs["input_ids"].size(1)
            self.tokens_processed += tokens_in_batch

            # Initial logging (vraiment une seule fois)
            if self.state.global_step == 0 and not self._initial_log_done:
                target_size = self.dataset_size if self.dataset_size else "unknown"
                dataset_name = (
                    self.data_loader.dataset_config.name
                    if self.data_loader and hasattr(self.data_loader, "dataset_config")
                    else "unknown"
                )
                self.logger.info(
                    f"Training started - Target: {target_size} tokens, "
                    f"Dataset: {dataset_name}"
                )
                self._initial_log_done = True

            # Get loss avec vérification de validité
            loss = super().training_step(model, inputs, return_loss)
            self._current_inputs = inputs

            # Sauvegarde des logs
            if not torch.isfinite(loss):
                self.logger.warning(
                    f"Non-finite loss detected at step {self.state.global_step}: {loss.item()}"
                )
                if self.state.global_step > 0:  
                    raise ValueError("Training stopped due to non-finite loss")
            if (
                self.state.global_step > 0
                and self.state.global_step % self.checkpoint_steps == 0
            ):
                self._save_checkpoint(self.state.global_step)

            return loss

        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            raise

    def _save_checkpoint(self, step):
        """Nouveau helper pour gérer la sauvegarde des checkpoints"""
        checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint-{step}")

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

            # Sauvegarder le modèle et le tokenizer une seule fois
            self.model.save_pretrained(checkpoint_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(checkpoint_path)
                self.logger.info(f"Tokenizer saved to {checkpoint_path}")

            # Sauvegarder l'état d'entraînement
            training_state = {
                "global_step": self.state.global_step,
                "tokens_processed": self.tokens_processed,
                "target_size": self.dataset_size,
                "log_history": self.state.log_history,
                "best_model_checkpoint": self.state.best_model_checkpoint,
                "training_time": self.state.total_flos,
                "epoch": self.state.epoch,
            }
            torch.save(
                training_state, os.path.join(checkpoint_path, "trainer_state.pt")
            )

            # Sauvegarder les métriques
            self._save_metrics_report(checkpoint_path)

            self.logger.info(f"Checkpoint saved to {checkpoint_path}")


class TrainingConfig:
    """Gère la configuration et l'exécution de l'entraînement"""

    def __init__(self, model_config, data_loader):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.data_loader = data_loader
        self.masking_handler = MaskingHandler(data_loader)  
        self.trainer = None
        self.model = None
        self.tokenizer = (
            data_loader.tokenizer
        )  

        # Setup des répertoires
        self.base_dir = "camembert-training"
        self.run_dir = self._setup_run_dir()
        self.device = self._setup_device()

        if self.model_config and self.model_config.config:
            self._initialize_model()

    def _setup_device(self) -> torch.device:
        """Configure le périphérique d'entraînement avec des paramètres optimaux"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_info = torch.cuda.get_device_properties(0)
            self.logger.info(
                f"Using GPU: {torch.cuda.get_device_name(0)} "
                f"({gpu_info.total_memory / 1024**3:.1f}GB)"
            )
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            return device
        elif torch.backends.mps.is_available():
            self.logger.warning("Using MPS (Apple Silicon). FP16 will be disabled.")
            return torch.device("mps")
        else:
            self.logger.warning(
                "No GPU detected, using CPU. Performance will be limited."
            )
            return torch.device("cpu")

    def _setup_run_dir(self) -> str:
        """Crée et configure le répertoire d'exécution de l'entraînement"""
        os.makedirs(self.base_dir, exist_ok=True)
        run_id = 0
        while True:
            run_dir = os.path.join(self.base_dir, f"cam_run{run_id}")
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
                return run_dir
            run_id += 1

    def _initialize_model(self):
        """Initialise et configure le modèle"""
        try:
            if not self.model_config.config:
                raise ValueError("Model configuration not initialized")

            self.model = RobertaForMaskedLM(self.model_config.config)
            self.model.to(self.device)
            self.logger.info(f"Model initialized on {self.device}")

        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def setup_trainer(self, training_args):
        try:
            if not self.data_loader.is_ready():
                raise ValueError("Dataset not loaded")

            # Utilisation de masking_handler pour le data_collator
            sample_batch = next(iter(self.data_loader.dataset))
            required_fields = {"input_ids", "attention_mask", "special_tokens_mask"}

            if not all(field in sample_batch for field in required_fields):
                missing_fields = required_fields - set(sample_batch.keys())
                raise ValueError(
                    f"Dataset stream missing required fields: {missing_fields}"
                )

            # Vérifier que le collate_fn fonctionne
            test_batch = self.masking_handler.data_collator(
                [sample_batch]
            ) 
            required_batch_fields = {"input_ids", "attention_mask", "labels"}

            if not all(k in test_batch for k in required_batch_fields):
                missing_batch_fields = required_batch_fields - set(test_batch.keys())
                raise ValueError(
                    f"Invalid batch structure after collate_fn. Missing: {missing_batch_fields}"
                )

            if not self.model:
                self._initialize_model()

            masking_monitor = MaskingMonitorCallback(
                tokenizer=self.tokenizer,
                expected_mlm_probability=self.masking_handler.mlm_probability,  
            )

            # Utiliser directement le dataset configuré
            self.trainer = CustomTrainer(
                data_loader=self.data_loader,
                model=self.model,
                args=training_args,
                train_dataset=self.data_loader.dataset,
                data_collator=self.masking_handler.data_collator,  
                callbacks=[masking_monitor],
                processing_class=self.tokenizer,
                masking_handler=self.masking_handler,  
            )

            # Garder la vérification finale
            self._verify_training_setup(masking_monitor)

        except Exception as e:
            self.logger.error(f"Error setting up trainer: {e}")
            raise

    def _verify_training_setup(self, masking_monitor: MaskingMonitorCallback):
        """Vérifie la configuration d'entraînement et le masquage"""
        try:
            # Test batch processing
            sample_batch = next(iter(self.data_loader.dataset))
            test_batch = self.masking_handler.data_collator(
                [sample_batch]
            ) 

            # Verification des champs requis
            required_keys = {"input_ids", "attention_mask", "labels"}
            if not all(k in test_batch for k in required_keys):
                raise ValueError(
                    f"Invalid batch structure. Missing: {required_keys - set(test_batch.keys())}"
                )

            # Permet de vérifier le masquage pendant le streaming des données
            stats = masking_monitor.analyze_batch(test_batch)
            self.logger.info(
                f"Setup verification:\n"
                f"- Current masking ratio: {stats['current_masking_ratio']:.2%}\n"
                f"- Expected ratio: {stats['expected_ratio']:.2%}"
            )

        except Exception as e:
            self.logger.error(f"Setup verification failed: {e}")
            raise

    def _calculate_training_parameters(self) -> dict:
        """Calcule les paramètres d'entraînement en utilisant les valeurs de model_config"""
        if not self.data_loader or not self.data_loader.dataset_size:
            raise ValueError("Dataset non initialisé")

        model_args = self.model_config.model_args

        cuda_available = torch.cuda.is_available()
        base_batch_size = model_args.batch_size if cuda_available else 16
        optimal_workers = 4 if cuda_available else 2

        # Dataset Size Analysis
        dataset_size_gb = self.data_loader.dataset_size * 4 / (1024**3)

        # Calcul du learning rate et du batch size
        gradient_acc = 16  
        if dataset_size_gb < 5:
            gradient_acc = 8
        elif dataset_size_gb < 20:
            gradient_acc = 12

        effective_batch_size = base_batch_size * gradient_acc

        base_lr = model_args.learning_rate
        batch_scale = (effective_batch_size / 256) ** 0.5
        learning_rate = base_lr * batch_scale

        # Calcul du nombre de batches et steps
        tokens_per_batch = base_batch_size * 512
        total_batches = self.data_loader.dataset_size // tokens_per_batch

        # Calcul du nombre d'updates par heure (dans les log)
        updates_per_hour = 3600 / (2.5 * (effective_batch_size / 256))
        total_steps = int(total_batches // gradient_acc)

        # Warmup steps
        warmup_steps = int(model_args.warmup_ratio * total_steps)

        # Affichage de l'état de l'entraînement dans les logs
        save_interval_hours = 1.0 if dataset_size_gb < 20 else 2.0
        save_steps = max(100, int(updates_per_hour * save_interval_hours))
        save_steps = min(save_steps, total_steps // 20)
        save_steps = max(save_steps, total_steps // 100)

        logging_steps = save_steps // 4

        # Affichage de l'information hardware
        if cuda_available:
            gpu_info = torch.cuda.get_device_properties(0)
            hardware_info = f"🚀 GPU ({gpu_info.name}, {gpu_info.total_memory / (1024**3):.1f}GB VRAM)"
        else:
            hardware_info = "🖥️ CPU (Test Local)"

        training_args = {
            "max_steps": total_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "gradient_accumulation_steps": gradient_acc,
            "per_device_train_batch_size": base_batch_size,
            "dataloader_num_workers": optimal_workers,
            "weight_decay": model_args.weight_decay,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "max_grad_norm": model_args.max_grad_norm,
            "lr_scheduler_type": "cosine_with_restarts",
        }

        log_message = (
            f"Configuration optimisée pour {dataset_size_gb:.1f}GB sur {hardware_info}:\n"
            f"- Batches totaux: {total_batches:,}\n"
            f"- Steps maximum: {total_steps:,}\n"
            f"- Learning rate: {learning_rate:.2e}\n"
            f"- Gradient accumulation: {gradient_acc}\n"
            f"- Batch size: {base_batch_size} (effectif: {effective_batch_size})\n"
            f"- Nombre de workers: {optimal_workers}\n"
            f"- Warmup steps: {warmup_steps:,}\n"
            f"- Schedule: Cosine avec restarts"
        )

        return {
            "training_args": training_args,
            "info": {
                "dataset_size_gb": dataset_size_gb,
                "total_batches": total_batches,
                "effective_batch_size": effective_batch_size,
                "estimated_hours": total_steps * (2.5 / 3600),
                "hardware_setup": hardware_info,
            },
            "log_message": log_message,
        }

    def start_training(
        self, output_dir: str, wandb_project: str, use_cuda: bool, fp16_training: bool
    ) -> str:
        try:
            if not self.model:
                return "❌ Model not initialized"
            cuda_available = torch.cuda.is_available() and use_cuda

            training_params = self._calculate_training_parameters()
            training_args = training_params["training_args"]

            cuda_available = torch.cuda.is_available() and use_cuda

            # Initialisation de wandb si GPU Sorbonne
            if cuda_available:
                run_name = os.path.basename(self.run_dir)
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    dir=self.run_dir,
                    config={
                        **training_args,
                        **training_params["info"],
                        "model_config": self.model.config.to_dict(),
                    },
                )

            # Configuration de l'entraînement
            args = TrainingArguments(
                output_dir=self.run_dir,
                **training_args,
                fp16=cuda_available and fp16_training,
                dataloader_pin_memory=cuda_available,
                report_to="wandb" if cuda_available else "none",
            )

            self.logger.info("Setting up trainer...")
            self.setup_trainer(args)

            self.logger.info("Starting training...")
            training_result = self.trainer.train()

            # Vérification de fin d'entraînement
            if training_result:
                self.logger.info("Training completed. Verifying final state...")
                training_finished = (
                    self.trainer.state.global_step >= self.trainer.args.max_steps
                )

                if training_finished:
                    self.logger.info(
                        "Training reached its target steps. Saving final state..."
                    )
                    # Sauvegarde finale du modèle
                    self.trainer.save_model()

                    # Log des métriques finales dans wandb
                    if wandb.run is not None:
                        try:
                            final_metrics = {
                                "final_loss": self.trainer.state.log_history[-1].get(
                                    "loss", None
                                ),
                                "total_steps": self.trainer.state.global_step,
                                "training_status": "completed",
                            }
                            wandb.log(final_metrics)
                            self.logger.info("Final metrics logged to wandb")
                        except Exception as e:
                            self.logger.error(
                                f"Error logging final metrics to wandb: {e}"
                            )
                        finally:
                            wandb.finish()
                            self.logger.info("Wandb run finished and closed")

            return "✅ Training completed successfully!"

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.logger.exception("Full traceback:")
            if wandb.run is not None:
                wandb.finish()
            return f"❌ Training error: {str(e)}"

    def _save_final_model(self):
        """Sauvegarde le modèle final avec gestion explicite du tokenizer"""
        try:
            if not self.trainer:
                return

            # Sauvegarder d'abord le modèle
            self.trainer.save_model()

            # Vérifier si wandb est actif
            if not wandb.run:
                return

            # Vérifier si c'est vraiment la fin du training
            training_finished = (
                self.trainer.state.global_step >= self.trainer.args.max_steps
            )

            if training_finished:
                try:
                    final_metrics = {
                        "final_loss": self.trainer.state.log_history[-1].get(
                            "loss", None
                        ),
                        "total_steps": self.trainer.state.global_step,
                        "training_finished": True,
                    }
                    wandb.log(final_metrics)
                except Exception as e:
                    self.logger.error(f"Error logging final metrics: {e}")
                finally:
                    wandb.finish()

        except Exception as e:
            self.logger.error(f"Error in _save_final_model: {e}")
            if wandb.run:
                wandb.finish()
            raise

    def stop_training(self):
        """Arrête l'entraînement et nettoie"""
        try:
            self.logger.info("Stopping training...")

            # Cleanup wandb (pour GPU)
            if wandb.run is not None:
                wandb.finish()
            if self.trainer:
                self.trainer.save_model()

            self.logger.info("Training stopped successfully!")

        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
            self.logger.exception("Exit failure")
            sys.exit(1)
