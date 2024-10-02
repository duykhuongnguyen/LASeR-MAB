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

    def score(self, query, response):
        """
        Compute the log likelihood score for a response using the reward model.

        Parameters:
        query (str): The input query or question.
        response (str): The model-generated response.

        Returns:
        float: The log likelihood score for the response.
        """
        inputs = self.tokenizer(query + response, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
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
        # Determine the device to load this model on (distribute across available devices if multi_gpu)
        device_for_model = devices[idx % len(devices)]

        # Load the model and tokenizer for the reward model
        model = AutoModelForCausalLM.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        
        # Create a RewardModel instance and add it to the list
        reward_models.append(RewardModel(name, model, tokenizer, device_for_model))
    
    return reward_models