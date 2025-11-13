#!/usr/bin/env bash
# Smoke-test launcher for GLADream distillation. Adjust paths before use.

set -euo pipefail

# Minimal resources so the run finishes quickly
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# Required paths (edit to match your environment)
STUDENT_BASE=${STUDENT_BASE:-"/path/to/fast_dllm_student"}
TEACHER_BASE=${TEACHER_BASE:-"/path/to/frozen_teacher"}
DATASET_PATH=${DATASET_PATH:-"/path/to/mini_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/distill_smoke"}

# Optional knobs for very small runs
MAX_STEPS=${MAX_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
ACC_STEPS=${ACC_STEPS:-1}
LOGGING_STEPS=${LOGGING_STEPS:-2}

python v2/train_scripts/distill.py \
  --model_name_or_path "${STUDENT_BASE}" \
  --teacher_model_name_or_path "${TEACHER_BASE}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --freeze_inherited_weights True \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${ACC_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps 0 \
  --teacher_visible_devices 0,1 \
  --student_visible_devices 2,3 \
  --hidden_loss_weight 1.0 \
  --logits_loss_weight 0.0 \
  --ce_loss_weight 1.0 \
  --disable_group_texts True \
  --dataloader_num_workers 0 \
  --report_to "none" \
  --seed 42
