#!/usr/bin/env bash
# Example launcher for GLADream distillation.
# Fill the variables below with actual checkpoints and dataset path before running.

set -euo pipefail

# Paths (edit these to match your environment)
STUDENT_BASE=${STUDENT_BASE:-"/path/to/fast_dllm_student"}
TEACHER_BASE=${TEACHER_BASE:-"/path/to/frozen_teacher"}
DATASET_PATH=${DATASET_PATH:-"/path/to/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/output_dir"}

# GPU assignment (teacher uses GPUs 0-1, student uses GPUs 2-3 by default)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

python v2/train_scripts/distill.py \
  --model_name_or_path "${STUDENT_BASE}" \
  --teacher_model_name_or_path "${TEACHER_BASE}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --freeze_inherited_weights True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 1e-6 \
  --warmup_steps 500 \
  --logging_steps 20 \
  --save_steps 500 \
  --save_total_limit 5 \
  --teacher_visible_devices 0,1 \
  --student_visible_devices 2,3 \
  --hidden_loss_weight 1.0 \
  --logits_loss_weight 0.1 \
  --ce_loss_weight 1.0 \
  --temperature 1.0 \
  --disable_group_texts True \
  --dataloader_num_workers 4 \
  --report_to "none" \
  --seed 42
