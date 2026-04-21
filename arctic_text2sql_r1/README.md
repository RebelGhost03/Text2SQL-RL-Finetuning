# Text2SQL BIRD Evaluation

## Overview

This repository provides a complete demo setup for evaluating Large Language Models on Text-to-SQL tasks. It specifically focuses on the BIRD (Benchmark for Instruction-following with Reasoning and Dialogue) benchmark to measure execution accuracy. The evaluation code in this project is derived from the Arctic-Text2SQL-R1 project, designed to support models such as Arctic-SQL-R1-7B and other open-source LLMs

Model on Hugging Face: [Ftel-Text2SQL](https://huggingface.co/Phucdh35/Ftel-Text2SQL)

### What's Inside
- **Data Preparation**: Scripts for generating data used in the evaluating process.
- **Evaluation**: Instructions for evaluating Arctic-Text2SQL-R1 on Bird benchmarks:
    - **BIRD**: Benchmark for Instruction-following with Reasoning and Dialogue
 
## Submission Configuration

### System Type
Single GPU inference using 1 × NVIDIA A100 80GB.

No multi-GPU, no distributed execution, and no external API usage.

Internet access is used only during setup for installing dependencies and downloading Hugging Face model weights. The evaluation phase runs fully offline.

### Runtime Environment
- CUDA 12.3
- Python ≥ 3.10
- Single-process execution on one GPU 

### Hardware Requirements
- 1 × A100 80GB GPU
- ≥ 16 CPU cores recommended
- ≥ 70 GB RAM recommended
- ≥ 100 GB disk space

## Important: Configuration Requirements

Before running the scripts, you must update the absolute folder paths within the scripts to match your local environment.

- **preprocess_data.sh**: Update the BIRD_DIR, DB_ROOT, and INDEX_ROOT variables to point to your dataset location.
- **run_evaluation.sh**: Update the PROCESSED_INPUT, GOLD_FILE, and DB_PATH variables to match your directory structure.


## Data Preprocess

### Manual Setup (Optional)

1. **Set Up Environment:**
   ```sh
   conda create -n arctic_process_data python=3.9.5
   conda activate arctic_process_data

   apt-get update
   apt-get install -y openjdk-11-jdk

   pip3 install func_timeout ijson tqdm pyserini==0.22.1 faiss-cpu torch numpy==1.24.3 nltk==3.8.1
   python3 nltk_downloader.py
   ```

2. **Download Datasets:**

Download the following datasets and extract them into a single directory. For the purposes of this tutorial we will assume they are all extracted into `/data/`.

- **BIRD Benchmark**
  - [Dev Set](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip)


3. **Generate Evaluation Input Data**

Here is an example, you should replace with your path

   ```sh
   # Build BM25 index for database values
   python3 data_preprocessing/build_contents_index.py \
   --db-root ./data/BIRD_DIR/dev_20240627/dev_databases \
   --index-root ./data/BIRD_DIR/dev_20240627/db_contents_index \
   --temp-dir ./data/BIRD_DIR/temp \
   --threads 16
   ```

   generate input file
   ```
    -i INPUT_JSON       path to input_data_file \
    -o OUTPUT_JSON      path to output_data_file \
    -d DB_PATH          path to the directory of databases \
    -t TABLES_JSON      path to tables definition JSON \
    -s SOURCE           data source name (e.g. "bird") \
    -m MODE             dataset mode (e.g. "dev") \
    -v VALUE_LIMIT_NUM  integer limit for values \
    -c DB_INDEX_PATH    path to db_content_index
   ```
   ```sh
   # Prepare input-output sequences, example here
   bash data_preprocessing/process_dataset.sh \
    -i ./data/BIRD_DIR/dev_20240627/dev.json \
    -o ./data/BIRD_DIR/dev_bird.json \
    -d ./data/BIRD_DIR/dev_20240627/dev_databases/ \
    -t ./data/BIRD_DIR/dev_20240627/dev_tables.json \
    -s bird \
    -m dev \
    -v 3 \
    -c ./data/BIRD_DIR/dev_20240627/db_contents_index
   ```

### Automated Preprocessing Script

We provide a comprehensive script preprocess_data.sh that automates environment creation, dataset downloading, indexing, and input generation.

Run Preprocessing:
   ```bash
   bash ./preprocess_data.sh
   ```

## Evaluation Process

### Evaluation Reproduction
You can easily reproduce our evaluation results as follows:

1. **Set Up Environment:**
   ```sh
   conda create -n arctic_eval python=3.9.5
   conda activate arctic_eval
   pip3 install vllm==0.10.0 func_timeout tqdm matplotlib nltk==3.8.1 sqlparse pandas
   ```

2. **Run Evaluation:**
Here is an example of evaluation a model, please replace the input paraments
   ```bash
   python3 bird_eval/eval_open_source_models.py \
   --models Snowflake/Arctic-Text2SQL-R1-7B \
   --input_file ./data/BIRD_DIR/dev_bird.json \
   --gold_file_path ./data/BIRD_DIR/dev_20240627/gold_dev_bird.json \
   --dp_path ./data/BIRD_DIR/dev_20240627/dev_databases \
   --self_consistency
   ```

### Automated Evaluation Script

Use the run_evaluation.sh script to set up the inference environment and execute the evaluation. Change the 

Run Evaluation:
   ```bash
   bash ./run_evaluation.sh
   ```

## Acknowledgments

This project incorporates code adapted from [OmniSQL – Synthesizing High-quality Text-to-SQL Data at Scale](https://github.com/RUCKBReasoning/OmniSQL) by RUCKBReasoning. And utilizes evaluation methodologies derived from the Arctic-Text2SQL-R1 [Arctic-Text2SQL-R1: Simple Rewards, Strong
Reasoning in Text-to-SQL](https://github.com/snowflakedb/ArcticTraining) project. We thank the authors for their contributions to the open-source community.
Please see their repository for license details and further usage guidelines. This repo builds on code from OmniSQL and Arctic-Text2SQL-R1 (Apache 2.0, see link above).
