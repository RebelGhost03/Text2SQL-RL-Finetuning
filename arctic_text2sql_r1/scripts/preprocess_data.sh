#!/bin/bash

# Exit on error
set -e

# 1. Set Up Environment
echo "Setting up preprocessing environment..."
conda create -n sql_preprocess python=3.9.5 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sql_preprocess

sudo apt-get update
sudo apt-get install -y openjdk-11-jdk

pip3 install spacy==3.7.6 func_timeout ijson tqdm pyserini==0.22.1 faiss-cpu torch numpy==1.24.3 nltk==3.8.1
python3 nltk_downloader.py

# 2. Download Datasets
echo "Downloading BIRD Dev Set..."
mkdir -p ./data
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip -O ./data/dev.zip
unzip ./data/dev.zip -d ./data/
mv ./data/dev_20240627/dev.json ./data/dev_20240627/gold_dev_bird.json

# Define Paths (Adjust these based on unzip results)
BIRD_DIR="./data/dev_20240627"
DB_ROOT="$BIRD_DIR/dev_databases"
INDEX_ROOT="$BIRD_DIR/db_contents_index"
TEMP_DIR="./temp_index"

# 3. Build BM25 index
echo "Building BM25 index..."
python3 data_preprocessing/build_contents_index.py \
--db-root "$DB_ROOT" \
--index-root "$INDEX_ROOT" \
--temp-dir "$TEMP_DIR" \
--threads 16

# 4. Generate Evaluation Input File
echo "Generating processed input JSON..."
bash data_preprocessing/process_dataset.sh \
 -i "$BIRD_DIR/gold_dev_bird.json" \
 -o "$BIRD_DIR/dev_bird_processed.json" \
 -d "$DB_ROOT/" \
 -t "$BIRD_DIR/dev_tables.json" \
 -s bird \
 -m dev \
 -v 2 \
 -c "$INDEX_ROOT"

echo "Preprocessing Complete!"