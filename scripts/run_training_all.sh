#!/bin/bash

# Ensure script is executed with bash
# Activate virtual environment if using one (uncomment if necessary)
# source /path/to/your/virtualenv/bin/activate

# Path to the config file
CONFIG_FILE="config.yaml"

# Ensure the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file ($CONFIG_FILE) not found!"
  exit 1
fi

# Install the required packages if not already installed (optional)
pip install -r requirements.txt

# Set the number of GPUs to use
export CUDA_VISIBLE_DEVICES=0,1  # Adjust this based on available GPUs, if multi-GPU support is needed

# List of datasets to train on (extracted from config.yaml)
DATASETS=("strategyqa" "gsm8k" "mmlu")

# Loop over each dataset and run the training script
for DATASET_NAME in "${DATASETS[@]}"; do
  echo "Starting training for dataset $DATASET_NAME..."
  
  # Run the Python training script for the current dataset
  python train_and_infer.py $DATASET_NAME
  
  # Check if the script ran successfully
  if [ $? -eq 0 ]; then
    echo "Training and inference completed successfully for dataset $DATASET_NAME!"
  else
    echo "There was an error during the training process for dataset $DATASET_NAME."
    exit 1
  fi
done