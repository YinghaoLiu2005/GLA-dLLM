#!/bin/bash

# Fast-dLLM v2 continuous pretraining with GLADream using torchrun (no DeepSpeed)
# Adjust variables below as needed.
export PYTORCH_ALLOC_CONF=expandable_segments:True

MASTER_PORT=${MASTER_PORT:-11001}
NUM_GPUS=${NUM_GPUS:-4}

MODEL_PATH=${MODEL_PATH:-/data/yinghaoliu/GLA-dLLM/trained_models/pretrain_gla_dream_from_fast_dllm_v2/checkpoint-1000}
BASE_MODEL_FOR_FREEZING=${BASE_MODEL_FOR_FREEZING:-/data/yinghaoliu/GLA-dLLM/trained_models/Fast_dLLM_v2_1.5B}
DATASET_PATH=${DATASET_PATH:-}
PREPROCESSED_DATASET_PATH=${PREPROCESSED_DATASET_PATH:-/home/yinghaoliu/preprocessed_data}
OUTPUT_DIR=${OUTPUT_DIR:-/data/yinghaoliu/GLA-dLLM/trained_models/pretrain_gla_dream_from_fast_dllm_v2}

EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
BLOCK_SIZE=${BLOCK_SIZE:-512}
BATCH_SIZE_PER_DEVICE=${BATCH_SIZE_PER_DEVICE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
BD_SIZE=${BD_SIZE:-32}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-100}
RUN_NAME=${RUN_NAME:-pretrain_gla_fastdllm_v2}
RESUME_PATH=${RESUME_PATH:-"/data/yinghaoliu/GLA-dLLM/trained_models/pretrain_gla_dream_from_fast_dllm_v2/checkpoint-1000"}

# Set to True to freeze inherited weights, only train randomly initialized modules
FREEZE_INHERITED=${FREEZE_INHERITED:-True}

PREPROCESSED_ARG=()
if [[ -n "${PREPROCESSED_DATASET_PATH}" ]]; then
    PREPROCESSED_ARG=(--preprocessed_dataset_path "${PREPROCESSED_DATASET_PATH}")
fi

DATASET_ARG=()
if [[ -n "${DATASET_PATH}" ]]; then
    DATASET_ARG=(--dataset_path "${DATASET_PATH}")
fi

set -e

echo "--- Starting GLADream Continued Pretraining (Golden Config) ---"
echo "Loading model from: ${MODEL_PATH}"
echo "Freezing weights based on: ${BASE_MODEL_FOR_FREEZING}"
echo "Outputting to: ${OUTPUT_DIR}"
echo "New Learning Rate: ${LR}"
echo "Gradient Checkpointing: false"
echo "Batch Size per Device: ${BATCH_SIZE_PER_DEVICE}"
echo "Gradient Accumulation Steps: ${GRAD_ACCUM_STEPS}"
echo "-------------------------------------------------------------"

# Export environment variables
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_BUILD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    /data/yinghaoliu/GLA-dLLM/v2/train_scripts/pretrain.py \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code True \
    "${DATASET_ARG[@]}" \
    "${PREPROCESSED_ARG[@]}" \
    --output_dir ${OUTPUT_DIR} \
    --conversation_template fast_dllm_v2 \
    --max_steps 16000 \
    --learning_rate ${LR} \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps ${WARMUP_STEPS} \
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
    --save_total_limit 2 \
    --gradient_checkpointing 1 \
    --bd_size ${BD_SIZE} \
    --ddp_find_unused_parameters True \
    --dataloader_drop_last True \
    --freeze_inherited_weights ${FREEZE_INHERITED} \
    --base_model_path_for_freezing ${BASE_MODEL_FOR_FREEZING} \
    --optim adamw_bnb_8bit \
    --max_grad_norm 1.0 \
    --overwrite_output_dir True \
    --resume_from_checkpoint ${RESUME_PATH}
    #--dataloader_num_workers 64 \




echo "Pretraining completed!"

