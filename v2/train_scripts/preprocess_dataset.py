#!/usr/bin/env python
# coding=utf-8
"""预处理数据集脚本：将原始数据转换为可直接用于训练的数据集"""

import os
import sys
import glob
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
from itertools import chain
import argparse

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'gla_dream'))

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the raw dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the preprocessed dataset")
    parser.add_argument("--block_size", type=int, default=512, help="Block size for text grouping")
    parser.add_argument("--disable_group_texts", action="store_true", help="Disable group_texts (use blocking instead)")
    parser.add_argument("--group_texts_batch_size", type=int, default=1000, help="Batch size for group_texts")
    parser.add_argument("--preprocessing_num_workers", type=int, default=16, help="Number of workers for preprocessing")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    args = parser.parse_args()
    
    print(f"[INFO] Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    
    model_max_length = tokenizer.model_max_length
    block_size = min(args.block_size, model_max_length)
    
    print(f"[INFO] Loading raw dataset from {args.dataset_path}...")
    parquet_files = glob.glob(os.path.join(args.dataset_path, '*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found at path: {args.dataset_path}")
    
    target_file = sorted(parquet_files)[0]
    raw_dataset = load_dataset('parquet', data_files={'train': target_file})['train']
    print(f"[INFO] Raw dataset size: {len(raw_dataset)} samples")
    
    # Tokenization
    print("[INFO] Tokenizing dataset...")
    text_column_name = "text"
    
    def tokenize_function(examples):
        if args.disable_group_texts:
            return tokenizer(
                examples[text_column_name],
                truncation=True,
                max_length=block_size,
                padding=False,
                return_special_tokens_mask=True
            )
        else:
            return tokenizer(
                examples[text_column_name],
                truncation=True,
                max_length=model_max_length,
                return_special_tokens_mask=True
            )
    
    tokenized_ds = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_dataset.column_names,  # 移除所有原始列
        desc="Running tokenizer on dataset",
    )
    print(f"[INFO] Tokenized dataset size: {len(tokenized_ds)} samples")
    
    # Group texts or blocking
    if args.disable_group_texts:
        print("[INFO] Applying blocking to each sample...")
        from lmflow.tokenization.hf_decoder_model import blocking
        
        def apply_blocking(examples):
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            padding_side = tokenizer.padding_side
            
            examples['labels'] = [ids.copy() for ids in examples['input_ids']]
            
            num_examples = len(examples['input_ids'])
            for i in range(num_examples):
                max_length = block_size
                pad_length = max_length - len(examples['input_ids'][i])
                
                if pad_length < 0:
                    for key in ['input_ids', 'attention_mask', 'labels']:
                        if key in examples:
                            examples[key][i] = examples[key][i][:max_length]
                else:
                    if padding_side == "right":
                        examples['input_ids'][i].extend([pad_token_id] * pad_length)
                        if 'attention_mask' in examples:
                            examples['attention_mask'][i].extend([0] * pad_length)
                        examples['labels'][i].extend([-100] * pad_length)
                    elif padding_side == "left":
                        examples['input_ids'][i] = [pad_token_id] * pad_length + examples['input_ids'][i]
                        if 'attention_mask' in examples:
                            examples['attention_mask'][i] = [0] * pad_length + examples['attention_mask'][i]
                        examples['labels'][i] = [-100] * pad_length + examples['labels'][i]
            
            return examples
        
        lm_dataset = tokenized_ds.map(
            apply_blocking,
            batched=True,
            batch_size=1,
            num_proc=args.preprocessing_num_workers,
            desc=f"Applying blocking (block_size={block_size})",
        )
    else:
        print("[INFO] Grouping texts into blocks...")
        
        def group_texts(examples):
            tokenizer_fields = ['input_ids', 'attention_mask', 'special_tokens_mask']
            available_fields = [k for k in tokenizer_fields if k in examples]
            
            if 'input_ids' not in available_fields:
                raise ValueError("input_ids field not found.")
            
            concatenated_examples = {}
            for k in available_fields:
                concatenated_examples[k] = list(chain(*examples[k]))
            
            total_length = len(concatenated_examples['input_ids'])
            for k in available_fields:
                if len(concatenated_examples[k]) != total_length:
                    raise ValueError(f"Field {k} length mismatch.")
            
            total_length = (total_length // block_size) * block_size
            
            if total_length == 0:
                return {k: [] for k in available_fields + ['labels']}
            
            result = {}
            for k in available_fields:
                result[k] = [
                    concatenated_examples[k][i : i + block_size] 
                    for i in range(0, total_length, block_size)
                ]
            
            result["labels"] = result["input_ids"].copy()
            return result
        
        lm_dataset = tokenized_ds.map(
            group_texts,
            batched=True,
            batch_size=args.group_texts_batch_size,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    
    print(f"[INFO] Processed dataset size: {len(lm_dataset)} samples")
    print(f"[INFO] Saving preprocessed dataset to {args.output_path}...")
    
    os.makedirs(args.output_path, exist_ok=True)
    lm_dataset.save_to_disk(args.output_path)
    
    print(f"[INFO] Dataset saved successfully!")
    print(f"[INFO] You can now load it in your training script with:")
    print(f"    --preprocessed_dataset_path {args.output_path}")

if __name__ == '__main__':
    main()

