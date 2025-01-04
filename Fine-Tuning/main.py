from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification
from safetensors.torch import load_file

# Download the model weights (SafeTensors file)
weights_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/model.safetensors",  # Path to your model weights
)

# Download the model configuration file
config_path = hf_hub_download(
    repo_id="CocoLng/CamemBERT-Gpt",
    filename="cam_run24/weights/config.json",  # Path to your config
)

# Load the weights from the .safetensors file
weights = load_file(weights_path)

# Load the model configuration
from transformers import AutoConfig
config = AutoConfig.from_pretrained(config_path)

# Load the model using the weights and configuration
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,  # We are not using a pre-trained model, so this is None
    config=config,  # Provide the configuration file
    state_dict=weights,  # Provide the weights dictionary
)

# Now you can use the model for inference
print(model)