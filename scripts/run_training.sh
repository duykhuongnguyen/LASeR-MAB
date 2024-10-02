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

# Get dataset name from the user
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME=$1

# Install the required packages if not already installed (optional)
pip install -r requirements.txt

# Set the number of GPUs to use
export CUDA_VISIBLE_DEVICES=1,2,3  # Adjust this based on available GPUs, if multi-GPU support is needed

# Run the Python training script for the specified dataset
python train_and_infer.py $DATASET_NAME

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "Training and inference completed successfully for dataset $DATASET_NAME!"
else
  echo "There was an error during the training process for dataset $DATASET_NAME."
  exit 1
fi