import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from transformers import Trainer, TrainingArguments
from transformers import RobertaForMaskedLM
import torch
import wandb
import os
from tqdm.auto import tqdm

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

class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        if 'tokenizer' in kwargs:
            kwargs.pop('tokenizer')  # Remove tokenizer from kwargs
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

class TrainingConfig:
    def __init__(self, model_config, data_loader):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.data_loader = data_loader
        self.trainer = None
        self.training_args = TrainerArguments()
        self.model = None

    def initialize_model(self) -> None:
        """Initialize RoBERTa model with config"""
        try:
            if not self.model_config.config:
                raise ValueError("Model configuration not initialized")

            self.model = RobertaForMaskedLM(self.model_config.config)
            self.logger.info("Model initialized successfully")

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
                report_to="wandb"
            )

        except Exception as e:
            self.logger.error(f"Error setting up training arguments: {e}")
            raise

    def setup_trainer(self) -> None:
        """Setup trainer with model and data collator"""
        try:
            if not self.model:
                self.initialize_model()

            self.trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.data_loader.dataset,
                data_collator=self.data_loader.data_collator
            )

        except Exception as e:
            self.logger.error(f"Error setting up trainer: {e}")
            raise

    def train(self) -> None:
        """Start the training process"""
        try:
            if not self.trainer:
                raise ValueError("Trainer not initialized")

            # Start training
            self.trainer.train()

            # Save the final model
            self.trainer.save_model()
            self.logger.info("Training completed successfully")

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise