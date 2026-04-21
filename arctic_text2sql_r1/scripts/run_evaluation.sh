#!/bin/bash

# Exit on error
set -e

# 1. Set Up Environment
echo "Setting up evaluation environment..."
conda create -n sql_eval python=3.9.5 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sql_eval

pip3 install vllm==0.10.0 func_timeout tqdm matplotlib nltk==3.8.1 sqlparse pandas

# Define Paths
MODEL_NAME="Phucdh35/Ftel-Text2SQL" # Replace with your model
PROCESSED_INPUT="./data/dev_20240627/dev_bird_processed.json"
GOLD_FILE="./data/dev_20240627/gold_dev_bird.json"
DB_PATH="./data/dev_20240627/dev_databases"

# 2. Run Evaluation
echo "Starting Evaluation for model: $MODEL_NAME"
python3 bird_eval/eval_open_source_models.py \
--models "$MODEL_NAME" \
--input_file "$PROCESSED_INPUT" \
--gold_file_path "$GOLD_FILE" \
--dp_path "$DB_PATH" \
--self_consistency

echo "Evaluation Finished!"