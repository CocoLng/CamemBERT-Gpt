from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import CamembertTokenizer
from safetensors.torch import load_file
from datasets import load_dataset
import os
import logging
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# Set up logging
logger = logging.getLogger(__name__)

# Download the model weights (SafeTensors file)
weights_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/model.safetensors",
)

# Download the model configuration file
config_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/config.json",  # Path to your config
)

# Load the weights from the .safetensors file
weights = load_file(weights_path)

# Load the model configuration
config = AutoConfig.from_pretrained(config_path)
config.num_labels = 3

# Load the model using the weights and configuration
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,  
    config=config, 
    state_dict=weights,
)

# Download the tokenizer files
tokenizer_json_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/tokenizer.json"
)

# Download the tokenizer configuration file
tokenizer_config_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/tokenizer_config.json"
)

# Download the vocabulary file
vocab_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/vocab.json"
)

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_json_path,
    vocab_file=vocab_path,
    config_file=tokenizer_config_path
)

# Changer pour tokenizer CamemBERT
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Charger le dataset "multi_nli" depuis Hugging Face
train_set = load_dataset("facebook/xnli", "fr", split="train")
test_set = load_dataset("facebook/xnli", "fr", split="test")
valid_set = load_dataset("facebook/xnli", "fr", split="validation")

# Tokenize the dataset
def tokenize_function(examples):
    tokens = tokenizer(
        examples["premise"], examples["hypothesis"],
        padding="max_length", truncation=True, return_tensors="pt"
    )
    tokens["labels"] = examples["label"] 
    return tokens

tokenized_train_set = train_set.map(tokenize_function, batched=True)
tokenized_test_set = test_set.map(tokenize_function, batched=True)
tokenized_valid_set = valid_set.map(tokenize_function, batched=True)

# Define the evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_set,
    eval_dataset=tokenized_valid_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()