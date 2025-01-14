import logging
import os
import sys

import torch
import wandb
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments

from data.masking_monitor import (
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
        """Étape d'entraînement avec surveillance et graphiques améliorés"""
        try:
            required_keys = {"input_ids", "attention_mask", "labels"}
            if not all(k in inputs for k in required_keys):
                raise ValueError(
                    f"Missing required keys: {required_keys - set(inputs.keys())}"
                )

            # Calcul efficace des tokens traités
            tokens_in_batch = inputs["input_ids"].size(0) * inputs["input_ids"].size(1)
            self.tokens_processed += tokens_in_batch

            # Initial logging
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
                self.save_checkpoint(self.state.global_step)

            return loss

        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            raise

    def _save_checkpoint(self, model=None, trial=None, metrics=None):
        """Surcharge de la méthode Trainer pour assurer la compatibilité"""
        self.save_checkpoint(self.state.global_step, metrics=metrics)


class TrainingConfig:
    """Gère la configuration et l'exécution de l'entraînement"""

    def __init__(self, model_config, data_loader):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.data_loader = data_loader
        self.masking_handler = MaskingHandler(data_loader)
        self.trainer = None
        self.model = None
        self.tokenizer = data_loader.tokenizer

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

            sample_batch = next(iter(self.data_loader.dataset))
            required_fields = {"input_ids", "attention_mask", "special_tokens_mask"}

            if not all(field in sample_batch for field in required_fields):
                missing_fields = required_fields - set(sample_batch.keys())
                raise ValueError(
                    f"Dataset stream missing required fields: {missing_fields}"
                )

            # Vérifier que le collate_fn fonctionne
            test_batch = self.masking_handler.data_collator([sample_batch])
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
            test_batch = self.masking_handler.data_collator([sample_batch])

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
        """Calcule les paramètres d'entraînement en fonction de la taille du dataset."""
        if not self.data_loader or not self.data_loader.dataset_size:
            raise ValueError("Dataset non initialisé")

        model_args = self.model_config.model_args

        # Utilise la taille du dataset en tokens
        total_tokens = self.data_loader.dataset_size

        # Configuration GPU et mémoire
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            base_batch_size = 64  # Optimisé pour ~95% utilisation GPU
            gradient_acc = 5      # Pour maintenir un effective batch size proche de l'original
            optimal_workers = min(4, os.cpu_count() or 1)
        else:
            base_batch_size = 8
            gradient_acc = 4
            optimal_workers = 1

        # Calcul du batch size effectif
        effective_batch_size = base_batch_size * gradient_acc

        peak_lr = 0.0007

        # Calcul du nombre de tokens par step
        tokens_per_step = effective_batch_size * 512  # 512 tokens par séquence

        # Calcul du nombre total de steps pour couvrir le dataset
        base_steps = max(1, total_tokens // tokens_per_step)

        # S'assure d'avoir un minimum de steps
        min_steps = 5000
        total_steps = max(base_steps, min_steps)

        # Limite le nombre total de steps à une valeur maximale si nécessaire
        total_steps = min(total_steps, 25000)

        # Configuration du warmup
        warmup_steps = min(10000, total_steps // 10)  # Par exemple, 10% des steps totaux

        # Configuration finale des arguments d'entraînement
        training_args = {
            "max_steps": total_steps,
            "learning_rate": peak_lr,  # Défini précédemment
            "warmup_steps": warmup_steps,
            "logging_steps": 250,
            "gradient_accumulation_steps": gradient_acc,
            "per_device_train_batch_size": base_batch_size,
            "dataloader_num_workers": optimal_workers,
            "weight_decay": model_args.weight_decay,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "max_grad_norm": model_args.max_grad_norm,
            "lr_scheduler_type": "polynomial",  # Explicitly using polynomial decay as per paper
        }

        # Préparation du message de log
        hardware_info = (
            f"🚀 GPU ({gpu_props.name}, {gpu_memory_gb:.1f}GB VRAM)"
            if torch.cuda.is_available()
            else "🖥️ CPU (Test Local)"
        )

        log_message = (
            f"Configuration optimisée pour {hardware_info}:\n"
            f"- Batch Size: {base_batch_size} (~95% GPU utilization)\n"
            f"- Gradient Accumulation: {gradient_acc}\n"
            f"- Effective Batch Size: {effective_batch_size}\n"
            f"- Peak Learning Rate: {peak_lr:.2e} (warmup: {warmup_steps:,} steps)\n"
            f"- Total Steps: {total_steps:,}\n"
            f"- Warmup Steps: {warmup_steps:,}\n"
            f"- Workers: {optimal_workers}\n"
            f"- Tokens per Step: {tokens_per_step:,}\n"
            f"- Scheduler: Polynomial decay"
        )

        return {
            "training_args": training_args,
            "info": {
                "total_tokens": total_tokens,
                "base_steps": base_steps,
                "effective_batch_size": effective_batch_size,
                "tokens_per_step": tokens_per_step,
                "gpu_memory": gpu_memory_gb if torch.cuda.is_available() else 0,
                "estimated_hours": total_steps * (2.5 / 3600),
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
                    # Appel à la méthode fusionnée save_model
                    self.trainer.save_model()  # Sauvegarde le modèle final

                    self.logger.info("Final model saved successfully")

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

    def stop_training(self):
        """Arrête l'entraînement et nettoie"""
        try:
            self.logger.info("Stopping training...")

            # Cleanup wandb (pour GPU)
            if wandb.run is not None:
                wandb.finish()
            if self.trainer:
                self.trainer.save_model()  # Utilise la méthode save_model fusionnée

            self.logger.info("Training stopped successfully!")

        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
            self.logger.exception("Exit failure")
            sys.exit(1)
