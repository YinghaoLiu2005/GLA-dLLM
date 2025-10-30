#!/bin/bash

# Fast-dLLM v2 training script with mixed initialization from Qwen3
# This script demonstrates how to use the integrated mixed initialization feature

echo "Starting Fast-dLLM v2 training with mixed initialization from Qwen3..."

deepspeed --master_port=11000 --num_gpus=4 v2/train_scripts/finetune.py \
    --model_name_or_path /data/yinghaoliu/Fast-dLLM/models/Qwen3-0.6B \
    --trust_remote_code 1 \
    --dataset_path /data/yinghaoliu/datasets/SFT/chat \
    --output_dir /data/yinghaoliu/Fast-dLLM/models/finetune_gla_dream_from_qwen3 \
    --conversation_template fast_dllm_v2 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --deepspeed v2/configs/ds_config_zero2_no_offload.json \
    --bf16 \
    --report_to wandb \
    --run_name finetune_mixed_init \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit 10 \
    --gradient_checkpointing 1 \
    --use_mixed_init 1 \
    --bd_size 32

echo "Training completed!"
