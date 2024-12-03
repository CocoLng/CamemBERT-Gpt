import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from transformers import RobertaForMaskedLM, Trainer, TrainerCallback, TrainingArguments
from typing import Optional
import threading

import wandb


class GradioTrainingCallback(TrainerCallback):
    def __init__(self, plot_component, metrics_component):
        self.logger = logging.getLogger(__name__)
        self.plot_component = plot_component
        self.metrics_component = metrics_component
        self.training_loss = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        try:
            # Mettre à jour la courbe d'apprentissage
            if "loss" in logs:
                self.training_loss.append(logs["loss"])
                self.steps.append(state.global_step)

                plt.figure(figsize=(10, 6))
                plt.plot(self.steps, self.training_loss)
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.title("Training Loss")

                # Update Gradio plot
                self.plot_component.update(value=plt.gcf())
                plt.close()

            # Mettre à jour les métriques actuelles
            current_metrics = {
                "loss": logs.get("loss", "N/A"),
                "learning_rate": logs.get("learning_rate", "N/A"),
                "epoch": logs.get("epoch", "N/A"),
                "step": state.global_step,
            }

            # Update Gradio metrics
            self.metrics_component.update(value=current_metrics)

        except Exception as e:
            self.logger.error(f"Error in callback: {e}")


@dataclass
class TrainerArguments:
    output_dir: str = "camembert-fr"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    logging_steps: int = 500
    save_steps: int = 10000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 4
    use_cuda: bool = True  # New parameter to control CUDA usage


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_training = False
        self.logger = logging.getLogger(__name__)

    def training_step(self, *args, **kwargs):
        if self.stop_training:
            self.logger.info("Training stopped by user request")
            raise KeyboardInterrupt
        return super().training_step(*args, **kwargs)

class TrainingConfig:
    def __init__(self, model_config, data_loader):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.data_loader = data_loader
        self.trainer: Optional[CustomTrainer] = None
        self.training_args = TrainerArguments()
        self.model = None
        self.device = self._setup_device()
        self.training_thread: Optional[threading.Thread] = None

    def _setup_device(self) -> torch.device:
        """Configure the device (CPU/GPU) for training"""
        if not self.training_args.use_cuda:
            self.logger.info("CUDA usage disabled by configuration")
            return torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )  # Convert to GB
            self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")

            # Optimize CUDA performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            return device
        else:
            self.logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

    def initialize_model(self) -> None:
        """Initialize RoBERTa model with config and move to appropriate device"""
        try:
            if not self.model_config.config:
                raise ValueError("Model configuration not initialized")

            self.model = RobertaForMaskedLM(self.model_config.config)
            self.model.to(self.device)

            # Log model device placement and memory usage
            if self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
                self.logger.info(
                    f"Model moved to GPU. Allocated memory: {memory_allocated:.2f}MB, "
                    f"Reserved memory: {memory_reserved:.2f}MB"
                )

            self.logger.info(
                "Model initialized successfully on device: " + str(self.device)
            )

        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def setup_training_arguments(self, **kwargs) -> None:
        """Setup training arguments with custom parameters"""
        try:
            # Update training arguments with provided kwargs
            for key, value in kwargs.items():
                if hasattr(self.training_args, key):
                    setattr(self.training_args, key, value)

            # Create HuggingFace training arguments
            run_name = f"training-run-{self.training_args.output_dir}-{wandb.util.generate_id()}"

            # Adjust batch size based on GPU memory if using CUDA
            if self.device.type == "cuda":
                gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )  # GB
                if gpu_memory >= 40:
                    self.training_args.per_device_train_batch_size *= 2
                    self.logger.info(
                        f"Increased batch size to {self.training_args.per_device_train_batch_size} "
                        f"due to large GPU memory"
                    )

            self.training_args = TrainingArguments(
                output_dir=self.training_args.output_dir,
                num_train_epochs=self.training_args.num_train_epochs,
                per_device_train_batch_size=self.training_args.per_device_train_batch_size,
                learning_rate=self.training_args.learning_rate,
                weight_decay=self.training_args.weight_decay,
                warmup_steps=self.training_args.warmup_steps,
                logging_steps=self.training_args.logging_steps,
                save_steps=self.training_args.save_steps,
                max_steps=self.training_args.max_steps,
                gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
                run_name=run_name,
                report_to="wandb",
                # GPU specific arguments
                fp16=self.device.type
                == "cuda",  # Enable mixed precision training on GPU
                dataloader_num_workers=4
                if self.device.type == "cuda"
                else 0,  # Use multiple workers on GPU
                dataloader_pin_memory=self.device.type
                == "cuda",  # Pin memory for faster data transfer to GPU
            )

        except Exception as e:
            self.logger.error(f"Error setting up training arguments: {e}")
            raise

    def setup_trainer(self, callback=None) -> None:
        """Setup trainer with model and data collator"""
        try:
            if not self.model:
                self.initialize_model()

            callbacks = []
            if callback:
                callbacks.append(callback)

            self.trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.data_loader.dataset,
                data_collator=self.data_loader.data_collator,
                callbacks=callbacks,
            )

        except Exception as e:
            self.logger.error(f"Error setting up trainer: {e}")
            raise

    def train(self) -> None:
        """Start the training process"""
        try:
            if not self.trainer:
                raise ValueError("Trainer not initialized")

            # Log GPU memory usage before training
            if self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
                self.logger.info(
                    f"Pre-training GPU memory - Allocated: {memory_allocated:.2f}MB, "
                    f"Reserved: {memory_reserved:.2f}MB"
                )

            # Start training
            self.trainer.train()

            # Save the final model
            self.trainer.save_model()
            self.logger.info("Training completed successfully")

            # Log final GPU memory usage
            if self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
                self.logger.info(
                    f"Post-training GPU memory - Allocated: {memory_allocated:.2f}MB, "
                    f"Reserved: {memory_reserved:.2f}MB"
                )

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
