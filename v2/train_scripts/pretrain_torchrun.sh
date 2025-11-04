#!/bin/bash

# Fast-dLLM v2 continuous pretraining with GLADream using torchrun (no DeepSpeed)
# Adjust variables below as needed.

MASTER_PORT=${MASTER_PORT:-11001}
NUM_GPUS=${NUM_GPUS:-4}

MODEL_PATH=${MODEL_PATH:-/data/yinghaoliu/Fast-dLLM/models/Fast_dLLM_v2_1.5B}
DATASET_PATH=${DATASET_PATH:-/data/yinghaoliu/datasets/fineweb-10BT/sample/10BT}
OUTPUT_DIR=${OUTPUT_DIR:-/data/yinghaoliu/Fast-dLLM/models/pretrain_gla_dream_from_fast_dllm_v2}

EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
BLOCK_SIZE=${BLOCK_SIZE:-512}
BATCH_SIZE_PER_DEVICE=${BATCH_SIZE_PER_DEVICE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
BD_SIZE=${BD_SIZE:-32}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-1000}
RUN_NAME=${RUN_NAME:-pretrain_gla_fastdllm_v2}

# Set to True to freeze inherited weights, only train randomly initialized modules
FREEZE_INHERITED=${FREEZE_INHERITED:-True}

set -e

echo "Starting Fast-dLLM v2 continuous pretraining with GLADream from Fast_dLLM_v2_1.5B (using torchrun)..."

# Export environment variables to avoid DeepSpeed CUDA extension compilation issues
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_BUILD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    /data/yinghaoliu/Fast-dLLM/v2/train_scripts/pretrain.py \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code True \
    --dataset_path ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --conversation_template fast_dllm_v2 \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio ${WARMUP_RATIO} \
    --disable_group_texts 1 \
    --block_size ${BLOCK_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_DEVICE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --bf16 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --validation_split_percentage 0 \
    --logging_steps ${LOGGING_STEPS} \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps ${SAVE_STEPS} \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit 2 \
    --gradient_checkpointing 1 \
    --bd_size ${BD_SIZE} \
    --parquet_max_files 1 \
    --ddp_find_unused_parameters False \
    --dataloader_drop_last True \
    --freeze_inherited_weights ${FREEZE_INHERITED}

echo "Pretraining completed!"

