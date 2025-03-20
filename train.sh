#!/bin/bash

source ../py39/bin/activate

# Check if GPU IDs are provided as arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <gpu_ids>"
  echo "Example: $0 0,1,2"
  exit 1
fi

# Get GPU IDs from arguments
GPU_IDS=$1

# Compute number of processes per node
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Print selected GPUs and number of processes
echo "Using GPUs: $GPU_IDS"
echo "Number of processes per node: $NUM_GPUS"

# Run the training script
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS train.py