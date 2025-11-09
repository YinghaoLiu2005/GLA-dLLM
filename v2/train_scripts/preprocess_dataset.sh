#!/bin/bash

# Helper script to preprocess datasets using preprocess_dataset.py.
# Allows overriding defaults via environment variables or CLI flags.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_SCRIPT="${SCRIPT_DIR}/preprocess_dataset.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "[ERROR] Cannot find preprocess_dataset.py next to this script." >&2
  exit 1
fi

MODEL_PATH=${MODEL_PATH:-/data/yinghaoliu/GLA-dLLM/trained_models/Fast_dLLM_v2_1.5B}
RAW_DATASET_PATH=${RAW_DATASET_PATH:-/data/yinghaoliu/datasets/fineweb-10BT/sample/10BT}
OUTPUT_PATH=${OUTPUT_PATH:-/home/yinghaoliu/preprocessed_data}
BLOCK_SIZE=${BLOCK_SIZE:-512}
DISABLE_GROUP_TEXTS=${DISABLE_GROUP_TEXTS:-0}
GROUP_TEXTS_BATCH_SIZE=${GROUP_TEXTS_BATCH_SIZE:-1000}
PREPROCESSING_NUM_WORKERS=${PREPROCESSING_NUM_WORKERS:-16}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
EXTRA_ARGS=("$@")

DISABLE_FLAG=""
if [[ "${DISABLE_GROUP_TEXTS}" == "1" ]]; then
  DISABLE_FLAG="--disable_group_texts"
fi

TRUST_FLAG=""
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_FLAG="--trust_remote_code"
fi

python "${PYTHON_SCRIPT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${RAW_DATASET_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --block_size "${BLOCK_SIZE}" \
  --group_texts_batch_size "${GROUP_TEXTS_BATCH_SIZE}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  ${DISABLE_FLAG} \
  ${TRUST_FLAG} \
  "${EXTRA_ARGS[@]}"
