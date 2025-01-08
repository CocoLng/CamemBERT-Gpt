import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    CamembertTokenizer,
    Trainer,
    TrainingArguments,
    AdamW,
    AutoModel,
)
from datasets import load_dataset
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import logging
import os



class CustomNLIModel(nn.Module):
    """
    Custom model with a classification head for NLI tasks.
    """

    def __init__(self, config, weights, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
                                pretrained_model_name_or_path=None,  # We are not using a pre-trained model, so this is None
                                config=config,  # Provide the configuration file
                                state_dict=weights,  # Provide the weights dictionary
                                 )

        hidden_size = config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability
        self.hidden_layer = nn.Linear(hidden_size, hidden_size // 2)  # Hidden layer
        self.activation = nn.ReLU()  # Non-linearity
        self.classifier = nn.Linear(hidden_size // 2, num_labels)  # Output layer for NLI (3 classes)

    # Define loss function (CrossEntropyLoss for logits)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass through the model.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_states = outputs[1]  # Assume the second output is the pooled hidden state

        x = self.dropout(hidden_states)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits if loss is None else (loss, logits)


class Finetune_NLI:
    """
    Fine-tune an NLI model for Natural Language Inference tasks.
    """

    def __init__(self, model_repo, weights_filename, config_filename, voca_filename, tokenizer_name="camembert-base"):
        """
        Initialize the Finetune_NLI class with model, config, and tokenizer information.
        """
        self.model_repo = model_repo
        self.voca_filename = voca_filename
        self.weights_filename = weights_filename
        self.config_filename = config_filename
        self.tokenizer_name = tokenizer_name
        self.tokenizer = CamembertTokenizer.from_pretrained(self.tokenizer_name)
        self.model = None
        self.config = None

    def load_data(self, dataset_name="facebook/xnli", language="fr", test_split=0.2):
        """
        Load and tokenize the dataset for NLI tasks.
        """
        logger.info("Loading dataset...")
        dataset = load_dataset(dataset_name, language, split="train")


        def tokenize_function(examples):
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        logger.info("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=test_split, shuffle=True).values()
        test_dataset, validation_dataset = test_dataset.train_test_split(test_size=0.5).values()

        return train_dataset, test_dataset, validation_dataset

    def finetune_model(self, train_dataset, test_dataset, validation_dataset, num_epochs=1, batch=8,
                       learning_rate=1e-5, num_labels=3):

        """
        Fine-tune the NLI model using Hugging Face's Trainer API.
        """
        logger.info("Loading model configuration and weights...")
        # Download weights and config

        weights_path = hf_hub_download(self.model_repo, self.weights_filename)
        config_path = hf_hub_download(self.model_repo, self.config_filename)

        # Load config and weights
        self.config = AutoConfig.from_pretrained(config_path)

        weights = load_file(weights_path)

        # Initialize the custom model
        self.model = CustomNLIModel(self.config,weights, num_labels)


        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=8,
            learning_rate=learning_rate,
            logging_dir="./logs",
            logging_steps=1,
            report_to="tensorboard",
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        # Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        # Initialize Trainer
        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, None),  # Custom optimizer, no scheduler

        )

        # Fine-tune the model
        logger.info("Starting training...")
        trainer.train()

        # Utiliser predict pour effectuer des prédictions sur le jeu de validation
        logger.info("Prédictions sur le jeu de validation:")
        predictions = trainer.predict(validation_dataset)

        # Extraire les logits et les labels
        logits, labels = predictions.predictions, predictions.label_ids

        # Calculer l'accuracy
        preds = torch.tensor(logits).argmax(dim=1)  # Prédictions finales
        labels = torch.tensor(labels)  # Convertir les labels en tensor
        correct = (preds == labels).sum().item()
        accuracy = correct / len(labels)

        # Afficher l'accuracy
        logger.info(f"Accuracy sur le jeu de validation: {accuracy:.4f}")
        # Evaluate the model on the validation dataset
        logger.info("Evaluation on validation dataset:")
        results = trainer.evaluate(validation_dataset)
        logger.info(f"Evaluation results: {results}")
        # Save the trained model and tokenizer
        logger.info(f"Saving trained model to {save_dir}...")
        torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.config.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info("Model and tokenizer saved successfully.")

        return trainer


# Usage Example
finetuner = Finetune_NLI(
    model_repo="CocoLng/CamemBERT-Gpt",
    weights_filename="cam_run24/weights/model.safetensors",
    config_filename="cam_run24/weights/config.json",
    voca_filename = "cam_run24/weights/vocab.json"
)


# Save the trained model and tokenizer
save_dir = "./save"
os.makedirs(save_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load and prepare the data
train_data, test_data, validation_data = finetuner.load_data()

# Fine-tune the model
finetuner.finetune_model(train_data, test_data, validation_data)

