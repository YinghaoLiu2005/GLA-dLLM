#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data/yinghaoliu/GLA-dLLM:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/data/yinghaoliu/tmp
export TEMP=/data/yinghaoliu/tmp
export TMP=/data/yinghaoliu/tmp
export HF_HOME="/data/yinghaoliu/hf_cache"
export HF_DATASETS_CACHE="/data/yinghaoliu/hf_cache/datasets"
export TRANSFORMERS_CACHE="/data/yinghaoliu/hf_cache/hub"

model_name_or_path=/data/yinghaoliu/GLA-dLLM/trained_models/Fast_dLLM_v2_1.5B
custom_model_path=/data/yinghaoliu/GLA-dLLM/BiDeltaDiff/models
dataset_path=/data/yinghaoliu/datasets/SFT/code
output_dir=/data/yinghaoliu/GLA-dLLM/trained_models/finetune_fast_dLLM_v2_1.5B_BKDA
deepspeed_args="--master_port=11000"
conversation_template=fast_dllm_v2


trust_remote_code=1

latest_checkpoint=""
if [ -d "${output_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -name "checkpoint-*" -type d | sort -V | tail -1)
    if [ -n "${latest_checkpoint}" ]; then
        echo "Found latest checkpoint: ${latest_checkpoint}"
    else
        echo "No checkpoint found in ${output_dir}"
        latest_checkpoint=""
    fi
else
    echo "Output directory ${output_dir} does not exist, training from scratch"
    latest_checkpoint=""
fi

resume_arg=""
if [ -n "${latest_checkpoint}" ]; then
    resume_arg="--resume_from_checkpoint ${latest_checkpoint}"
fi

cmd="deepspeed ${deepspeed_args} \
  v2/train_scripts/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --custom_model_path ${custom_model_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --deepspeed v2/configs/ds_config_zero2_no_offload.json \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --save_total_limit 2 \
    --gradient_checkpointing 1 \
    --dataset_num_shards 8 \
    --dataset_shard_index 0 \
    "

echo $cmd
eval $cmd