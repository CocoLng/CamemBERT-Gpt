import logging
from typing import Optional
import os
import torch
import wandb
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM
from .masking_monitor import MaskingMonitorCallback

class CustomTrainer(Trainer):
    def __init__(self, data_loader=None, checkpoint_steps=5000, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        self.dataset_size = data_loader._dataset_size if data_loader else None
        self.checkpoint_steps = checkpoint_steps
        self.tokens_processed = 0
        self._initial_log_done = False
        
        # Utilisation de processing_class au lieu de tokenizer
        self.processing_class = kwargs.pop('processing_class', None)
        if self.processing_class is None and data_loader is not None:
            self.processing_class = data_loader.tokenizer
        
        # Setup directory structure
        self.base_dir = kwargs['args'].output_dir
        self.weights_dir = os.path.join(self.base_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Override save_steps in training arguments
        kwargs['args'].save_steps = self.checkpoint_steps
        
        # Ensure dataset is properly set
        if data_loader and 'train_dataset' not in kwargs:
            kwargs['train_dataset'] = data_loader.dataset
        
        # Mise à jour pour utiliser le data_loader mis à jour
        if data_loader:
            self.data_loader = data_loader
            self.dataset_size = data_loader._dataset_size
            if 'train_dataset' not in kwargs or kwargs['train_dataset'] is None:
                kwargs['train_dataset'] = data_loader.dataset  # Utilise le dataset du data_loader

        super().__init__(**kwargs)
        
        # Save initial configuration
        self._save_model_info(self.base_dir)

    

    def _save_model_info(self, directory: str):
        """Save comprehensive model information"""
        try:
            info_path = os.path.join(directory, "model_info.txt")
            with open(info_path, "w") as f:
                # Dataset information
                f.write("=== Dataset Information ===\n")
                if self.dataset_size:
                    f.write(f"Total tokens: {self.dataset_size}\n")
                    f.write(f"Approximate size in GB: {self.dataset_size * 4 / (1024**3):.2f}\n")
                f.write("\n")

                # Model architecture
                f.write("=== Model Architecture ===\n")
                config_dict = self.model.config.to_dict()
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Training parameters
                f.write("=== Training Parameters ===\n")
                training_params = {
                    'learning_rate': self.args.learning_rate,
                    'batch_size': self.args.per_device_train_batch_size,
                    'max_steps': self.args.max_steps,
                    'warmup_steps': self.args.warmup_steps,
                    'weight_decay': self.args.weight_decay,
                    'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
                    'save_steps': self.args.save_steps,
                    'logging_steps': self.args.logging_steps,
                    'mlm_probability': self.data_loader.mlm_probability if self.data_loader else None
                }
                for key, value in training_params.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            self.logger.error(f"Error saving model info: {e}")

    def training_step(self, model, inputs, return_loss=True):
        """Training step with optimal monitoring and performance"""
        try:
            # Validate inputs
            required_keys = {'input_ids', 'attention_mask', 'labels'}
            if not all(k in inputs for k in required_keys):
                raise ValueError(f"Missing required keys: {required_keys - set(inputs.keys())}")

            # Calcul efficace des tokens traités
            tokens_in_batch = inputs['input_ids'].size(0) * inputs['input_ids'].size(1)
            self.tokens_processed += tokens_in_batch

            # Initial logging (vraiment une seule fois)
            if self.state.global_step == 0 and not self._initial_log_done:
                self.logger.info(
                    f"Training started - Target: {self.dataset_size:,} tokens, "
                    f"Dataset: {self.data_loader.dataset_config.name}"
                )
                self._initial_log_done = True

            # Get loss avec vérification de validité
            loss = super().training_step(model, inputs, return_loss)

            # Passer les inputs au callback via un attribut temporaire
            self._current_inputs = inputs

            if not torch.isfinite(loss):
                self.logger.warning(
                    f"Non-finite loss detected at step {self.state.global_step}: {loss.item()}"
                )
                if self.state.global_step > 0:  # Évite l'arrêt au premier step
                    raise ValueError("Training stopped due to non-finite loss")

            # Gestion optimisée des checkpoints et logging
            if (self.state.global_step > 0 and 
                self.state.global_step % self.checkpoint_steps == 0):
                
                # Calcul de la progression
                progress = min(100, self.tokens_processed / self.dataset_size * 100)
                
                # Calcul de la moyenne des losses récentes (utilise l'historique existant)
                recent_losses = [log.get('loss', 0) for log in self.state.log_history[-10:]]
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
                
                # Log concis mais informatif
                self.logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Step: {self.state.global_step} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"Tokens: {self.tokens_processed:,}/{self.dataset_size:,}"
                )
                
                # Sauvegarde checkpoint si nécessaire
                checkpoint_dir = os.path.join(
                    self.base_dir, 
                    f"checkpoint-{self.state.global_step}"
                )
                if not os.path.exists(checkpoint_dir):
                    self._save_comprehensive_checkpoint(checkpoint_dir)

            return loss

        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            raise


    def _save_comprehensive_checkpoint(self, checkpoint_dir: str):
        """Save comprehensive checkpoint with stream handling"""
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 1. Save model state and configuration
            super().save_model(checkpoint_dir)
            
            # 2. Save processing_class (tokenizer)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(checkpoint_dir)

            # 3. Save training state with stream information
            training_state = {
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'tokens_processed': self.tokens_processed,
                'target_size': self.dataset_size,
                'log_history': self.state.log_history,
                'best_model_checkpoint': self.state.best_model_checkpoint
            }
            torch.save(training_state, os.path.join(checkpoint_dir, "trainer_state.pt"))
            
            # 4. Save detailed metrics
            self._save_metrics_report(checkpoint_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise

    def _save_metrics_report(self, checkpoint_dir: str):
        """Save detailed metrics report"""
        try:
            report_path = os.path.join(checkpoint_dir, "metrics_report.txt")
            with open(report_path, "w") as f:
                f.write("=== Training Report ===\n\n")
                
                # Current state
                f.write(f"Global Step: {self.state.global_step}\n")
                f.write(f"Epoch: {self.state.epoch}\n\n")
                
                # Learning rates
                f.write("Learning Rates:\n")
                for group_id, group in enumerate(self.optimizer.param_groups):
                    f.write(f"Group {group_id}: {group['lr']}\n")
                f.write("\n")
                
                # Loss history
                f.write("Recent Loss History:\n")
                recent_logs = [log for log in self.state.log_history if 'loss' in log][-10:]
                for log in recent_logs:
                    f.write(f"Step {log.get('step', 'N/A')}: {log.get('loss', 'N/A')}\n")
                
        except Exception as e:
            self.logger.error(f"Error saving metrics report: {e}")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with enhanced final weights handling"""
        try:
            # Sauvegarder le modèle avec la méthode parent
            super().save_model(output_dir, _internal_call=_internal_call)

            save_dir = output_dir if output_dir else self.weights_dir
            
            # Sauvegarder le processing_class (tokenizer)
            if self.processing_class is not None:
                self.logger.info(f"Saving tokenizer to {save_dir}")
                self.processing_class.save_pretrained(save_dir)
            else:
                self.logger.warning("No processing_class available to save!")

            # Si c'est la sauvegarde finale
            if output_dir is None or (output_dir == self.args.output_dir and not _internal_call):
                self.logger.info("Saving final weights...")
                self._save_model_info(self.weights_dir)
                super().save_model(self.weights_dir, _internal_call=True)
                self._save_metrics_report(self.weights_dir)
                
                # Ne pas gérer wandb ici - laissez TrainingConfig le faire
                
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

class TrainingConfig:
    """Manages training configuration and execution"""
    def __init__(self, model_config, data_loader):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.data_loader = data_loader
        self.trainer = None
        self.model = None
        self.tokenizer = data_loader.tokenizer  # Important: Stockage explicite du tokenizer
        
        # Setup directories and device
        self.base_dir = "camembert-training"
        self.run_dir = self._setup_run_dir()
        self.device = self._setup_device()
        
        # Initialize model if configuration is ready
        if self.model_config and self.model_config.config:
            self._initialize_model()

    def _setup_device(self) -> torch.device:
        """Configure training device with optimal settings"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_info = torch.cuda.get_device_properties(0)
            self.logger.info(
                f"Using GPU: {torch.cuda.get_device_name(0)} "
                f"({gpu_info.total_memory / 1024**3:.1f}GB)"
            )
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            return device
        elif torch.backends.mps.is_available():
            self.logger.warning("Using MPS (Apple Silicon). FP16 will be disabled.")
            return torch.device("mps")
        else:
            self.logger.warning("No GPU detected, using CPU. Performance will be limited.")
            return torch.device("cpu")

    def _setup_run_dir(self) -> str:
        """Create and setup training run directory"""
        os.makedirs(self.base_dir, exist_ok=True)
        run_id = 0
        while True:
            run_dir = os.path.join(self.base_dir, f"cam_run{run_id}")
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
                os.makedirs(os.path.join(run_dir, "weights"))
                return run_dir
            run_id += 1

    def _initialize_model(self):
        """Initialize and configure the model"""
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

            # Nous gardons les vérifications nécessaires mais utilisons directement le dataset
            sample_batch = next(iter(self.data_loader.dataset))
            required_fields = {'input_ids', 'attention_mask', 'special_tokens_mask'}
            
            if not all(field in sample_batch for field in required_fields):
                missing_fields = required_fields - set(sample_batch.keys())
                raise ValueError(f"Dataset stream missing required fields: {missing_fields}")
                
            # Vérifier que le collate_fn fonctionne
            test_batch = self.data_loader.data_collator([sample_batch])
            required_batch_fields = {'input_ids', 'attention_mask', 'labels'}
            
            if not all(k in test_batch for k in required_batch_fields):
                missing_batch_fields = required_batch_fields - set(test_batch.keys())
                raise ValueError(f"Invalid batch structure after collate_fn. Missing: {missing_batch_fields}")

            if not self.model:
                self._initialize_model()

            masking_monitor = MaskingMonitorCallback(
                tokenizer=self.tokenizer,
                expected_mlm_probability=self.data_loader.mlm_probability
            )

            # Utiliser directement le dataset configuré
            self.trainer = CustomTrainer(
                data_loader=self.data_loader,
                model=self.model,
                args=training_args,
                train_dataset=self.data_loader.dataset,
                data_collator=self.data_loader.data_collator,
                callbacks=[masking_monitor],
                processing_class=self.tokenizer
            )

            # Garder la vérification finale
            self._verify_training_setup(masking_monitor)
                    
        except Exception as e:
            self.logger.error(f"Error setting up trainer: {e}")
            raise


    def _verify_training_setup(self, masking_monitor: MaskingMonitorCallback):
        """Verify training setup and masking configuration"""
        try:
            # Test batch processing
            sample_batch = next(iter(self.data_loader.dataset))
            test_batch = self.data_loader.data_collator([sample_batch])
            
            # Verify required tensors
            required_keys = {'input_ids', 'attention_mask', 'labels'}
            if not all(k in test_batch for k in required_keys):
                raise ValueError(f"Invalid batch structure. Missing: {required_keys - set(test_batch.keys())}")

            # Verify masking using the monitor
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
        """
        Calculate optimized training parameters for OSCAR/mOSCAR with adaptive configurations.
        Key features:
        - Fixed optimal batch size with gradient accumulation for effective batch size control
        - RoBERTa-style learning rate schedule with proper warmup
        - Dynamic steps and epochs based on dataset size
        - Optimized cosine schedule with restarts
        """
        if not self.data_loader or not self.data_loader.dataset_size:
            raise ValueError("Dataset non initialisé")

        # Fixed optimal batch size as per RoBERTa recommendations
        cuda_available = torch.cuda.is_available()
        base_batch_size = 75 if cuda_available else 16
        optimal_workers = 4 if cuda_available else 2

        # Dataset Size Analysis
        dataset_size_gb = self.data_loader.dataset_size * 4 / (1024**3)
        
        # Adaptive Gradient Accumulation for effective batch size control
        # Smaller datasets need more accumulation for better convergence
        if dataset_size_gb < 5:
            gradient_acc = 8  # Effective batch size: 600
        elif dataset_size_gb < 20:
            gradient_acc = 12  # Effective batch size: 900
        else:
            gradient_acc = 16  # Effective batch size: 1200
            
        # Calculate effective batch size
        effective_batch_size = base_batch_size * gradient_acc
        
        # RoBERTa-style Learning Rate Configuration
        base_lr = 6e-4  # RoBERTa base learning rate
        # Scale learning rate based on effective batch size
        batch_scale = (effective_batch_size / 256) ** 0.5
        learning_rate = base_lr * batch_scale
        
        # Calculate total batches and steps
        tokens_per_batch = base_batch_size * 512
        total_batches = self.data_loader.dataset_size // tokens_per_batch
        
        # Adaptive number of epochs based on dataset size
        if dataset_size_gb < 5:
            num_epochs = 5  # More epochs for smaller datasets
        elif dataset_size_gb < 20:
            num_epochs = 3
        else:
            num_epochs = 2  # Fewer epochs for larger datasets
            
        # Calculate total steps based on real data size
        updates_per_epoch = total_batches // gradient_acc
        total_steps = updates_per_epoch * num_epochs
        
        # RoBERTa standard 6% warmup
        warmup_steps = int(0.06 * total_steps)
        
        # Adaptive save frequency based on training progress
        updates_per_hour = 3600 / (2.5 * (effective_batch_size / 256))
        save_interval_hours = 1.0 if dataset_size_gb < 20 else 2.0
        save_steps = max(100, int(updates_per_hour * save_interval_hours))
        
        # Ensure reasonable number of saves (between 20 and 100 saves total)
        save_steps = min(save_steps, total_steps // 20)
        save_steps = max(save_steps, total_steps // 100)
        
        # More frequent logging for better monitoring
        logging_steps = save_steps // 4

        # Hardware info string
        if cuda_available:
            gpu_info = torch.cuda.get_device_properties(0)
            hardware_info = f"🚀 GPU ({gpu_info.name}, {gpu_info.total_memory / (1024**3):.1f}GB VRAM)"
        else:
            hardware_info = "🖥️ CPU (Test Local)"

        training_args = {
            'num_train_epochs': num_epochs,
            'max_steps': total_steps,
            'learning_rate': learning_rate,
            'warmup_steps': warmup_steps,
            'save_steps': save_steps,
            'logging_steps': logging_steps,
            'gradient_accumulation_steps': gradient_acc,
            'per_device_train_batch_size': base_batch_size,
            'dataloader_num_workers': optimal_workers,
            'weight_decay': 0.01,  # RoBERTa paper value
            'adam_beta1': 0.9,
            'adam_beta2': 0.98,  # RoBERTa paper value
            'max_grad_norm': 1.0,
            'lr_scheduler_type': "cosine_with_restarts",  # Cosine schedule with restarts
        }

        return {
            'training_args': training_args,
            'info': {
                'dataset_size_gb': dataset_size_gb,
                'total_batches': total_batches,
                'effective_batch_size': effective_batch_size,
                'estimated_hours': total_steps * (2.5 / 3600),
                'hardware_setup': hardware_info
            },
            'log_message': (
                f"Configuration optimisée pour {dataset_size_gb:.1f}GB sur {hardware_info}:\n"
                f"- Batches totaux: {total_batches:,}\n"
                f"- Epochs: {num_epochs}\n"
                f"- Steps maximum: {total_steps:,}\n"
                f"- Learning rate: {learning_rate:.2e}\n"
                f"- Gradient accumulation: {gradient_acc}\n"
                f"- Batch size: {base_batch_size} (effectif: {effective_batch_size})\n"
                f"- Nombre de workers: {optimal_workers}\n"
                f"- Warmup steps: {warmup_steps:,} ({(warmup_steps/total_steps)*100:.1f}%)\n"
                f"- Sauvegarde tous les {save_steps:,} steps (~{save_interval_hours:.1f}h)\n"
                f"- Schedule: Cosine avec restarts"
            )
        }

    def start_training(self, output_dir: str, wandb_project: str, use_cuda: bool, fp16_training: bool) -> str:
        try:
            if not self.model:
                return "❌ Model not initialized"

            # Get training parameters
            training_params = self._calculate_training_parameters()
            training_args = training_params['training_args']
            
            cuda_available = torch.cuda.is_available() and use_cuda
            
            # Initialize wandb with detailed config
            if cuda_available:
                run_name = os.path.basename(self.run_dir)
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    dir=self.run_dir,
                    config={
                        **training_args,
                        **training_params['info'],
                        'model_config': self.model.config.to_dict()
                    }
                )

            # Create training arguments
            args = TrainingArguments(
                output_dir=self.run_dir,
                **training_args,
                fp16=cuda_available and fp16_training,
                dataloader_pin_memory=cuda_available,  # Set this only here
                report_to="wandb" if cuda_available else "none",
            )

            self.logger.info("Setting up trainer...")
            self.setup_trainer(args)
            
            self.logger.info("Starting training...")
            self.trainer.train()
            
            return "✅ Training completed successfully!"

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.logger.exception("Full traceback:")
            return f"❌ Training error: {str(e)}"
    def _save_final_model(self):
        """Save final model with explicit tokenizer handling"""
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
                self.trainer.state.global_step >= self.trainer.args.max_steps or
                self.trainer.state.epoch >= self.trainer.args.num_train_epochs
            )

            if training_finished:
                try:
                    final_metrics = {
                        "final_loss": self.trainer.state.log_history[-1].get("loss", None),
                        "total_steps": self.trainer.state.global_step,
                        "total_epochs": self.trainer.state.epoch,
                        "training_finished": True
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
        """Stop training and cleanup"""
        try:
            self.logger.info("Stopping training...")
            
            # Cleanup wandb
            if wandb.run is not None:
                wandb.finish()
            
            # Save final model state
            if self.trainer:
                self.trainer.save_model()
            
            # Force exit
            import sys
            sys.exit(0)
            
        except Exception as e:
            self.logger.error(f"Error during training stop: {e}")
            sys.exit(1)