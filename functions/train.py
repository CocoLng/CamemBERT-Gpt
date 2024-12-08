import logging
from typing import Optional
import os
import shutil
import torch
import wandb
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM
from .masking_monitor import MaskingMonitorCallback

class CustomTrainer(Trainer):
    def __init__(self, data_loader=None, checkpoint_steps=20, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        self.dataset_size = data_loader._dataset_size if data_loader else None
        self.checkpoint_steps = checkpoint_steps
        
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
        """Training step with checkpoint management"""
        try:
            # Validate inputs
            required_keys = {'input_ids', 'attention_mask', 'labels'}
            if not all(k in inputs for k in required_keys):
                raise ValueError(f"Missing required keys in inputs: {required_keys - set(inputs.keys())}")

            # Use parent's training step
            loss = super().training_step(model, inputs, return_loss)
            
            # Create comprehensive checkpoint at save_steps
            if self.state.global_step > 0 and self.state.global_step % self.checkpoint_steps == 0:
                checkpoint_dir = os.path.join(self.base_dir, f"checkpoint-{self.state.global_step}")
                if not os.path.exists(checkpoint_dir):  # Évite les sauvegardes multiples
                    self._save_comprehensive_checkpoint(checkpoint_dir)
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            raise


    def _save_comprehensive_checkpoint(self, checkpoint_dir: str):
        """Save comprehensive checkpoint with all necessary information"""
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 1. Save model state and configuration
            super().save_model(checkpoint_dir)
            
            # 2. Save processing_class (tokenizer)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(checkpoint_dir)

            # Save optimizer and scheduler states
            optimizer_states = {
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'scaler': self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else None
            }
            torch.save(optimizer_states, os.path.join(checkpoint_dir, "optimizer.pt"))

            # Save training state
            training_state = {
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'log_history': self.state.log_history,
                'best_model_checkpoint': self.state.best_model_checkpoint
            }
            torch.save(training_state, os.path.join(checkpoint_dir, "trainer_state.pt"))
            
            # Save metrics report
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

            # Si c'est la sauvegarde finale ou un checkpoint
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
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(self.weights_dir)
                self._save_metrics_report(self.weights_dir)
                    
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

    def setup_trainer(self, training_args: TrainingArguments):
        """Setup trainer with monitoring and validation"""
        try:
            if not self.data_loader.is_ready():
                raise ValueError("Dataset not loaded")

            if not self.model:
                self._initialize_model()

            # Setup masking monitor
            masking_monitor = MaskingMonitorCallback(
                tokenizer=self.tokenizer,  # Utilisation du tokenizer stocké
                expected_mlm_probability=self.data_loader.mlm_probability
            )

            # Setup trainer with explicit tokenizer
            self.trainer = CustomTrainer(
                data_loader=self.data_loader,
                model=self.model,
                args=training_args,
                train_dataset=self.data_loader.dataset,
                data_collator=self.data_loader.data_collator,
                callbacks=[masking_monitor],
                processing_class=self.tokenizer  # Passage explicite du tokenizer
            )

            # Verify setup with the monitor
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

    def _verify_masking_stats(self, stats) -> bool:
        """Verify masking statistics are within acceptable range"""
        if not stats:
            return False
            
        tolerance = 0.02  # 2% tolerance
        expected = self.data_loader.mlm_probability
        actual = stats['current_masking_ratio']
        
        within_tolerance = abs(actual - expected) <= tolerance
        
        if not within_tolerance:
            self.logger.warning(
                f"Masking ratio {actual:.2%} outside tolerance range "
                f"[{expected-tolerance:.2%}, {expected+tolerance:.2%}]"
            )
            
        return within_tolerance

    def start_training(self, output_dir: str, num_train_epochs: int, batch_size: int,
                      learning_rate: float, weight_decay: float, warmup_steps: int,
                      gradient_accumulation: int, wandb_project: str,
                      use_cuda: bool, fp16_training: bool, num_workers: int,
                      max_steps: int) -> str:
        """Start training with specified configuration"""
        try:
            if not self.model:
                return "❌ Model not initialized"

            # Initialize wandb
            wandb.init(project=wandb_project, 
                    name=f"training-run-{os.path.basename(output_dir)}")

            # Determine if FP16 can be used
            can_use_fp16 = (torch.cuda.is_available() and use_cuda and fp16_training)
            if fp16_training and not can_use_fp16:
                self.logger.warning("FP16 requested but not available. Falling back to FP32.")

            # Create training arguments with explicit output directory
            training_args = TrainingArguments(
                output_dir=self.run_dir,
                max_steps=max_steps,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                gradient_accumulation_steps=gradient_accumulation,
                fp16=can_use_fp16,
                dataloader_num_workers=num_workers if torch.cuda.is_available() else 0,
                dataloader_pin_memory=torch.cuda.is_available(),
                report_to="wandb",
                logging_steps=min(500, max_steps // 20),
                save_steps=min(5000, max_steps // 10)
            )

            # Setup and start training with explicit tokenizer handling
            self.setup_trainer(training_args)
            self.trainer.train()
            
            # Save final model and tokenizer
            self._save_final_model()
            
            return "✅ Training completed successfully!"

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            if hasattr(self, 'trainer') and self.trainer is not None:
                self._save_final_model()  # Tentative de sauvegarde même en cas d'erreur
            return f"❌ Training error: {str(e)}"
        
    def _save_final_model(self):
        """Save final model with explicit tokenizer handling"""
        try:
            # Sauvegarder le modèle
            self.trainer.save_model()
            
            # Sauvegarder explicitement le tokenizer dans le dossier weights
            weights_dir = os.path.join(self.run_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            self.tokenizer.save_pretrained(weights_dir)
            
            self.logger.info(f"Model and tokenizer saved successfully in {weights_dir}")
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}")
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