#!/bin/bash
source /workspace/.venv/bin/activate
cd /workspace/EasyR1

export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

set -x

MODEL_PATH="/model"  # replace it with your local file path

python3 -m verl.trainer.main \
    config=/workspace/EasyR1/examples/config.yaml \
    data.train_files=/workspace/dataset/test.json \
    data.val_files=/workspace/dataset/tes1.json \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=8 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.2 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.reward.reward_function_kwargs='{"max_response_length":2048,"overlong_buffer_length":1024,"overlong_penalty_factor":1.0}' \
    trainer.experiment_name=qwen3_coder_30b_a3b_text2sql_grpo \
    trainer.n_gpus_per_node=8
