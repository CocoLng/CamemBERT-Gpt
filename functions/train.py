import logging
from dataclasses import dataclass
from transformers import Trainer, TrainingArguments, TrainerCallback, RobertaForMaskedLM
import matplotlib.pyplot as plt
import torch
import wandb
from typing import List, Optional
from .masking_monitor import MaskingMonitorCallback
import os
import shutil


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
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
                self.steps.append(state.global_step)
                
                plt.figure(figsize=(10, 6))
                plt.plot(self.steps, self.training_loss)
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                
                # Update Gradio plot
                self.plot_component.update(value=plt.gcf())
                plt.close()
                
            # Mettre à jour les métriques actuelles
            current_metrics = {
                'loss': logs.get('loss', 'N/A'),
                'learning_rate': logs.get('learning_rate', 'N/A'),
                'epoch': logs.get('epoch', 'N/A'),
                'step': state.global_step
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
    """Custom trainer with enhanced error handling and monitoring"""
    
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        if 'tokenizer' in kwargs:
            kwargs.pop('tokenizer')
        super().__init__(**kwargs)
        
        # Modifier la fréquence des checkpoints
        self.args.save_steps = 5000  # Changement de 10000 à 5000
        
        # Créer les dossiers nécessaires
        self.weights_dir = os.path.join(self.args.output_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Sauvegarder la configuration initiale
        self._save_model_info(self.args.output_dir)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step avec logging des losses"""
        try:
            # Vérifier les clés nécessaires
            required_keys = {'input_ids', 'attention_mask', 'labels'}
            if not all(k in inputs for k in required_keys):
                missing = required_keys - set(inputs.keys())
                raise ValueError(f"Missing required keys in inputs: {missing}")

            # Mettre les tenseurs sur le bon device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Calcul de la loss
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Enregistrer la loss dans le fichier de log
            if self.state.global_step % self.args.logging_steps == 0:
                self._log_loss(loss.item())

            # À chaque checkpoint, sauvegarder les informations
            if self.state.global_step % self.args.save_steps == 0:
                checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
                self._save_model_info(checkpoint_dir, loss.item())
                
            return loss
            
        except Exception as e:
            self.logger.error(f"Error in training_step: {e}")
            raise

    def _log_loss(self, loss: float):
        """Enregistre la loss dans un fichier"""
        log_path = os.path.join(self.args.output_dir, "loss_history.txt")
        try:
            with open(log_path, "a") as f:
                f.write(f"Step {self.state.global_step}: {loss}\n")
        except Exception as e:
            self.logger.error(f"Error logging loss: {e}")

    def _save_model_info(self, directory: str, current_loss: float = None):
        """Sauvegarde les informations du modèle"""
        try:
            info_path = os.path.join(directory, "model_info.txt")
            with open(info_path, "w") as f:
                # Architecture du modèle
                f.write("=== Architecture du Modèle ===\n")
                config_dict = self.model.config.to_dict()
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")

                # Paramètres d'entraînement
                f.write("\n=== Paramètres d'Entraînement ===\n")
                f.write(f"Learning rate: {self.args.learning_rate}\n")
                f.write(f"Batch size: {self.args.per_device_train_batch_size}\n")
                f.write(f"Nombre d'epochs: {self.args.num_train_epochs}\n")
                f.write(f"Warmup steps: {self.args.warmup_steps}\n")
                f.write(f"Weight decay: {self.args.weight_decay}\n")
                f.write(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}\n")
                
                if current_loss is not None:
                    f.write(f"Current loss: {current_loss}\n")
                
                # État actuel
                f.write(f"\nStep actuel: {self.state.global_step}\n")
                f.write(f"Epoch actuelle: {self.state.epoch}\n")
                
        except Exception as e:
            self.logger.error(f"Error saving model info: {e}")

    def save_model(self, output_dir=None):
        """Override pour sauvegarder les poids avec les informations"""
        # Sauvegarde standard du modèle
        super().save_model(output_dir)
        
        # Sauvegarder une copie dans le dossier weights avec les informations
        if output_dir:
            try:
                # Copier les poids
                shutil.copytree(output_dir, self.weights_dir, dirs_exist_ok=True)
                # Sauvegarder les informations finales
                self._save_model_info(self.weights_dir)
            except Exception as e:
                self.logger.error(f"Error saving final weights: {e}")

class TrainingConfig:
    def __init__(self, model_config, data_loader):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.data_loader = data_loader
        self.trainer = None
        # Initialisation des training_args avec des valeurs par défaut
        self.training_args = TrainerArguments(
            output_dir="camembert-fr",
            use_cuda=True  # valeur par défaut
        )
        self.model = None
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Configure the device (CPU/GPU) for training"""
        if not self.training_args.use_cuda:
            self.logger.info("CUDA usage disabled by configuration")
            return torch.device("cpu")
            
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
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
                self.logger.info(f"Model moved to GPU. Allocated memory: {memory_allocated:.2f}MB, "
                               f"Reserved memory: {memory_reserved:.2f}MB")
            
            self.logger.info("Model initialized successfully on device: " + str(self.device))

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
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if gpu_memory >= 40:  
                    self.training_args.per_device_train_batch_size *= 2
                    self.logger.info(f"Increased batch size to {self.training_args.per_device_train_batch_size} "
                                   f"due to large GPU memory")

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
                fp16=self.device.type == "cuda",  # Enable mixed precision training on GPU
                dataloader_num_workers=4 if self.device.type == "cuda" else 0,  # Use multiple workers on GPU
                dataloader_pin_memory=self.device.type == "cuda",  # Pin memory for faster data transfer to GPU
            )

        except Exception as e:
            self.logger.error(f"Error setting up training arguments: {e}")
            raise

    def setup_trainer(self, callback: Optional[TrainerCallback] = None) -> None:
        """Configuration complète du trainer avec monitoring"""
        try:
            # Vérifications préliminaires
            if not self.data_loader.is_ready():
                raise ValueError("Dataset non chargé. Chargez le dataset avant l'entraînement.")

            if not self.model:
                self.initialize_model()

            # Configuration du monitoring de masquage
            self.masking_monitor = MaskingMonitorCallback(
                tokenizer=self.data_loader.tokenizer,
                expected_mlm_probability=self.data_loader.mlm_probability
            )

            # Préparation des callbacks
            callbacks: List[TrainerCallback] = [self.masking_monitor]
            if callback:
                callbacks.append(callback)

            # Création du trainer
            self.trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.data_loader.dataset,
                data_collator=self.data_loader.data_collator,
                callbacks=callbacks
            )

            # Vérification finale
            self._verify_training_setup()

            self.logger.info("✅ Trainer configuré avec succès!")

        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la configuration du trainer: {e}")
            raise

    def _verify_training_setup(self) -> None:
        """Vérifie la configuration complète de l'entraînement"""
        try:
            # Vérifier un batch d'exemple
            sample_batch = next(iter(self.data_loader.dataset))
            collated_batch = self.data_loader.data_collator([sample_batch])
            
            # Vérifier la présence des tenseurs requis
            required_keys = {'input_ids', 'attention_mask', 'labels'}
            if not all(k in collated_batch for k in required_keys):
                raise ValueError(f"Batch invalide. Clés manquantes: {required_keys - set(collated_batch.keys())}")

            # Vérifier le masquage sur le batch d'exemple
            self.masking_monitor.analyze_batch(collated_batch)
            stats = self.masking_monitor.get_stats_dict()  # Utilisation du bon nom de méthode
            
            self.logger.info(
                f"Vérification du setup:\n"
                f"- Ratio de masquage test: {stats['current_masking_ratio']:.2%}\n"
                f"- Ratio attendu: {stats['expected_ratio']:.2%}"
            )

        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification du setup: {e}")
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
                self.logger.info(f"Pre-training GPU memory - Allocated: {memory_allocated:.2f}MB, "
                               f"Reserved: {memory_reserved:.2f}MB")

            # Start training
            self.trainer.train()

            # Save the final model
            self.trainer.save_model()
            self.logger.info("Training completed successfully")

            # Log final GPU memory usage
            if self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
                self.logger.info(f"Post-training GPU memory - Allocated: {memory_allocated:.2f}MB, "
                               f"Reserved: {memory_reserved:.2f}MB")

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise