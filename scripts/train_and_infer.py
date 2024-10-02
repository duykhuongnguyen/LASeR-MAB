import sys
import torch
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from utils.config_loader import load_config
from utils.dataset_manager import DatasetManager
from model.lora_model import setup_lora_model
from model.response_generator import ResponseGenerator
from utils.preference_pair_generator import PreferencePairGenerator
from utils.linucb import LinUCB
from utils.llm_trainer import LLMTrainer
from utils.reward_model import load_reward_models

# Load configuration from config.yaml
config = load_config("config.yaml")

# Get dataset name from command-line argument
if len(sys.argv) != 2:
    print("Usage: python train_and_infer.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]

if dataset_name not in config['datasets']:
    print(f"Error: Dataset '{dataset_name}' not found in config.yaml")
    sys.exit(1)

# Load LLaMA-3-8B model with LoRA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = config["model_name"]
model, tokenizer = setup_lora_model(model_name)
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Initialize DatasetManager to load datasets and generate prompts
dataset_paths = {dataset_name: config["datasets"][dataset_name]}
test_size = config["training"]["test_size"]
dev_size = config["training"]["dev_size"]
prompt_type = "reasoning"

dataset_manager = DatasetManager(dataset_paths, test_size=test_size, dev_size=dev_size, prompt_type=prompt_type)
breakpoint()


# Reward models from RewardBench
multi_gpu = torch.cuda.device_count() > 1  # Use multi-GPU if available
reward_model_names = config["reward_models"]
reward_models = load_reward_models(reward_model_names, device, multi_gpu=multi_gpu)

# Initialize response generator with 0-shot reasoning prompt format
n_responses = config["training"]["n_responses"]
temperature = config["training"]["temperature"]
response_generator = ResponseGenerator(model, tokenizer, prompt_type=prompt_type, temperature=temperature)

# Initialize preference pair generator
trainer = LLMTrainer(model, Adam(model.parameters(), lr=config["training"]["learning_rate"]))

# LinUCB setup for MAB
K = len(reward_model_names)
linucb = LinUCB(d=model.module.config.hidden_size if torch.cuda.device_count() > 1 else model.config.hidden_size, K=K)

# Training parameters
M = config["training"]["iterations"]
batch_size = config["training"]["batch_size"]
eval_threshold = config["training"]["eval_threshold"]

# Main training loop for the selected dataset
print(f"Training on {dataset_name} dataset")

for iteration in range(M):
    total_loss = 0
    train_dataset = dataset_manager.get_train_data(dataset_name)
    for i in range(0, len(train_dataset), batch_size):
        batch = train_dataset[i:i+batch_size]
        queries = [example['question'] for example in batch]

        # Get embeddings and select a reward model using LinUCB
        context = response_generator.get_embedding(queries)
        selected_rm = linucb.select_arm(context)

        # Generate responses
        responses = response_generator.generate_responses(queries, n_responses=n_responses)

        # Generate preference pairs from the selected reward model
        preference_pair_gen = PreferencePairGenerator(reward_models[selected_rm])
        preference_pairs = preference_pair_gen.generate_preference_pairs(responses)

        # Perform a training step
        loss = trainer.train_step(preference_pairs, model)
        linucb.update(selected_rm, context, -loss)
        total_loss += loss

    print(f"Iteration {iteration + 1}, Loss: {total_loss / len(train_dataset)}")

    # Check convergence
    if abs(total_loss / len(train_dataset)) < eval_threshold:
        print("Converged.")
        break

# After training, evaluate on the test set
print(f"Evaluating {dataset_name} on the test set")
test_dataset = dataset_manager.get_test_data(dataset_name)
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i+batch_size]
    queries = [example['question'] for example in batch]

    # Generate responses for inference
    responses = response_generator.generate_responses(queries, n_responses=n_responses)

    # Compute rewards using all reward models
    for rm in reward_models:
        rm_rewards = [rm.score(query, response) for query, response in responses]
        print(f"Rewards from {rm.name}: {rm_rewards}")