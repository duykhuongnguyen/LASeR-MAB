import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel:
    def __init__(self, name, model, tokenizer, device):
        """
        Initializes a Reward Model for scoring responses.

        Parameters:
        name (str): The name of the reward model.
        model (transformers.PreTrainedModel): The reward model loaded from Hugging Face.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the reward model.
        device (torch.device): The device (CPU or GPU) on which the reward model will run.
        """
        self.name = name
        self.model = model.to(device)  # Load the model onto the specified device (GPU or CPU)
        self.tokenizer = tokenizer
        self.device = device
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def score(self, prompt, response):
        """
        Compute the log likelihood score for a response using the reward model.

        Parameters:
        prompt (str): The formatted prompt presented to the model.
        response (str): The model-generated response.

        Returns:
        float: The log likelihood score for the response.
        """
        text = f"{prompt}{response}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        log_likelihood = -outputs.loss.item()
        return log_likelihood

def load_reward_models(model_names, device, multi_gpu=False):
    """
    Load multiple reward models by their names and distribute them across available GPUs if needed.

    Parameters:
    model_names (list of str): List of reward model names.
    device (torch.device): The primary device to use (GPU or CPU).
    multi_gpu (bool): If True, distribute reward models across multiple GPUs.

    Returns:
    list of RewardModel: List of loaded reward models.
    """
    reward_models = []
    
    # Get the list of available devices (GPUs) if multi_gpu is enabled
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if multi_gpu and torch.cuda.is_available() else [device]
    
    for idx, name in enumerate(model_names):
        # Determine repository id and display name if an alias is provided
        if " " in name:
            model_id, display_name = name.split(" ", 1)
        else:
            model_id = name
            display_name = name

        # Determine the device to load this model on (distribute across available devices if multi_gpu)
        device_for_model = devices[idx % len(devices)]

        # Load the model and tokenizer for the reward model
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Create a RewardModel instance and add it to the list
        reward_models.append(RewardModel(display_name, model, tokenizer, device_for_model))
    
    return reward_models
