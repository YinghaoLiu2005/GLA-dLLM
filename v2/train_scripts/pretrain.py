#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""Continuous pretraining entry script using GLADream initialized from Fast-dLLM v2.

This script replaces the original Dream model with GLADream, reusing compatible
weights from an existing Fast_dLLM_v2_1.5B checkpoint and randomly initializing
new modules. It then launches the `pretrainer` pipeline for continued pretraining.
"""

import sys
import os
import tempfile
import pathlib
from datasets import load_dataset, Dataset as HuggingFaceDataset
import glob
import pandas as pd
from itertools import chain
from transformers import Trainer, TrainingArguments
# Add the src directory to Python path for lmflow module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

# Fix for Accelerator.unwrap_model() compatibility issue
try:
    from accelerate import Accelerator
    original_unwrap_model = Accelerator.unwrap_model

    def patched_unwrap_model(self, model, **kwargs):
        # Remove the problematic parameter that doesn't exist in older versions
        kwargs.pop('keep_torch_compile', None)
        return original_unwrap_model(self, model, **kwargs)

    Accelerator.unwrap_model = patched_unwrap_model
except Exception as e:
    print(f"Warning: Could not patch Accelerator.unwrap_model: {e}")

from transformers import HfArgumentParser, AutoConfig
import torch

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

# Import GLADream for mixed/partial initialization
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'gla_dream'))
try:
    from gla_dream.modeling_gla_dream import GLADreamModel
    from gla_dream.configuration_gla_dream import GLADreamConfig
    GLA_DREAM_AVAILABLE = True
except ImportError:
    GLA_DREAM_AVAILABLE = False
    print("Warning: GLA Dream not available. Pretraining with Dream is disabled.")


def _freeze_inherited_weights(gla_model, fast_dllm_model_path, trust_remote_code):
    """
    Freeze weights that were loaded from Fast-dLLM, only allow training of
    randomly initialized modules.
    
    Strategy: Load the original Fast-dLLM state dict and freeze any parameters
    in GLADream that match (shape and name prefix).
    """
    try:
        from transformers import AutoModelForCausalLM
        import torch.nn as nn
        
        # Load original Fast-dLLM state dict to identify inherited weights
        print(f"[Freeze] Loading Fast-dLLM state dict from {fast_dllm_model_path}...")
        fast_dllm_model = AutoModelForCausalLM.from_pretrained(
            fast_dllm_model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,  # Use float32 for comparison
        )
        fast_dllm_state = fast_dllm_model.state_dict()
        del fast_dllm_model  # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get GLADream state dict
        gla_state = gla_model.state_dict()
        
        # Count frozen and trainable parameters
        frozen_count = 0
        trainable_count = 0
        
        print("[Freeze] Identifying inherited vs new parameters...")
        for name, param in gla_model.named_parameters():
            # Check if this parameter exists in Fast-dLLM with matching shape
            is_inherited = False
            if name in fast_dllm_state:
                if param.shape == fast_dllm_state[name].shape:
                    is_inherited = True
            
            if is_inherited:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1
        
        print(f"[Freeze] Frozen {frozen_count} inherited parameters, {trainable_count} parameters remain trainable.")
        
    except Exception as e:
        print(f"[Freeze] Warning: Could not freeze inherited weights: {e}")
        print("[Freeze] Falling back to training all parameters. This may use more memory.")


# ... (前面的代码保持不变) ...
import tempfile
import pathlib
from datasets import load_dataset, Dataset as HuggingFaceDataset
import glob
import pandas as pd

# 导入原生 Trainer 和 TrainingArguments
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

# Add the src directory to Python path for lmflow module
# ... (the rest of the imports and _freeze_inherited_weights function remain the same) ...

def main():
    # 1. 解析参数 (这部分保持不变)
    pipeline_name = "pretrainer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # 2. 构建模型
    print("[Pretrain] Using GLADream initialized from Fast_dLLM_v2_1.5B...")
    base_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    gla_config = GLADreamConfig(
        vocab_size=getattr(base_config, 'vocab_size', 151936),
        hidden_size=getattr(base_config, 'hidden_size', 1024),
        num_hidden_layers=getattr(base_config, 'num_hidden_layers', 24),
        num_attention_heads=getattr(base_config, 'num_attention_heads', 16),
        intermediate_size=getattr(base_config, 'intermediate_size', 2816),
        max_position_embeddings=getattr(base_config, 'max_position_embeddings', 32768),
        bd_size=getattr(data_args, 'bd_size', getattr(base_config, 'bd_size', 32)),
    )
    gla_model = GLADreamModel.from_fast_dllm_pretrained(
        fast_dllm_model_path=model_args.model_name_or_path,
        gla_config=gla_config,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    
    # 冻结权重
    if getattr(model_args, 'freeze_inherited_weights', False):
        _freeze_inherited_weights(gla_model, model_args.model_name_or_path, model_args.trust_remote_code)

    # 3. 【全新部分】加载和处理数据
    print("\n[INFO] Loading and tokenizing data directly using transformers standard flow...")
    
    # 检查是否提供了预处理的数据集路径
    preprocessed_dataset_path = getattr(data_args, 'preprocessed_dataset_path', None)
    
    if preprocessed_dataset_path and os.path.exists(preprocessed_dataset_path):
        print(f"[INFO] Loading preprocessed dataset from {preprocessed_dataset_path}...")
        from datasets import load_from_disk, DatasetDict

        dataset_on_disk = load_from_disk(preprocessed_dataset_path)
        if isinstance(dataset_on_disk, DatasetDict):
            if 'train' not in dataset_on_disk:
                raise ValueError("Preprocessed DatasetDict must contain a 'train' split.")
            dataset_on_disk = dataset_on_disk['train']

        # Materialize the dataset into RAM to avoid repeated disk I/O during training.
        print("[INFO] Materializing preprocessed dataset into RAM...")
        def _identity(batch):
            return batch

        train_dataset = dataset_on_disk.map(
            _identity,
            batched=True,
            batch_size=1000,
            keep_in_memory=True,
            load_from_cache_file=False,
            desc="Materializing dataset into RAM",
        )
        del dataset_on_disk

    train_dataset.set_format(type="torch")

    print(f"[INFO] Preprocessed dataset ready in RAM with {len(train_dataset)} samples.")
    dataset_num_bytes = None
    try:
        dataset_num_bytes = train_dataset._data.nbytes
    except AttributeError:
        dataset_num_bytes = None
    if dataset_num_bytes:
        approx_gb = dataset_num_bytes / (1024 ** 3)
        print(f"[INFO] Approximate RAM footprint: {approx_gb:.2f} GiB.")
        print(f"[INFO] Skipping on-the-fly preprocessing steps.")
    else:
        # 加载原始数据到内存
        parquet_files = glob.glob(os.path.join(data_args.dataset_path, '*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found at path: {data_args.dataset_path}")
        target_parquet_file = parquet_files[0]
        raw_dataset = load_dataset('parquet', data_files={'train': target_parquet_file})['train']

        # 从 data_args 获取 block_size，而不是 pipeline_args
        # 处理 block_size 为 None 的情况，并确保不超过 tokenizer 的最大长度
        model_max_length = tokenizer.model_max_length
        
        # 定义文本列名
        text_column_name = "text" # 确保这是你正确的列名
        
        if data_args.block_size is None:
            block_size = model_max_length
            if block_size > 1024:
                print(
                    "Warning: The chosen tokenizer supports a `model_max_length` that is"
                    " longer than the default `block_size` value of 1024. "
                    "If you would like to use a longer `block_size` up to `tokenizer.model_max_length` "
                    "you can override this default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > model_max_length:
                print(
                    f"Warning: The block_size passed ({data_args.block_size}) is larger than the maximum length "
                    f"for the model ({model_max_length}). Using block_size={model_max_length}."
                )
                block_size = model_max_length
            else:
                block_size = data_args.block_size
        
        # 执行分词和打包
        print("[INFO] Tokenizing dataset...")
        num_proc = getattr(data_args, 'preprocessing_num_workers', None)

        # 检查是否需要禁用group_texts
        disable_group_texts = getattr(data_args, 'disable_group_texts', False)

        def tokenize_function(examples):
            # 根据disable_group_texts决定是否truncate
            if disable_group_texts:
                # 当disable_group_texts=True时，在tokenization阶段就truncate到block_size
                return tokenizer(
                    examples[text_column_name],
                    truncation=True,
                    max_length=block_size,  # 直接truncate到block_size
                    padding=False,  # 不padding，后续在blocking中处理
                    return_special_tokens_mask=True
                )
            else:
                # 当disable_group_texts=False时，允许更长的序列，后续group_texts会处理
                return tokenizer(
                    examples[text_column_name],
                    truncation=True,
                    max_length=model_max_length,
                    return_special_tokens_mask=True
                )

        # 优化：移除所有原始列，减少后续处理的数据量
        tokenized_ds = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=raw_dataset.column_names,  # 移除所有原始列，不仅仅是text列
            desc="Running tokenizer on dataset",
            load_from_cache_file=not getattr(data_args, 'overwrite_cache', False),
        )

        # 定义优化的group_texts函数
        def group_texts(examples):
            # 将所有文本拼接在一起
            # 在 batched=True 的情况下，examples 中的每个值都是列表的列表
            # 例如: examples['input_ids'] = [[1,2,3], [4,5,6], ...]
            # 我们需要将其展平为: [1,2,3,4,5,6,...]
            # 然后将展平后的序列切分成固定大小的块
            
            # 获取所有存在的字段（只处理 tokenizer 返回的字段）
            tokenizer_fields = ['input_ids', 'attention_mask', 'special_tokens_mask']
            available_fields = [k for k in tokenizer_fields if k in examples]
            
            if not available_fields or 'input_ids' not in available_fields:
                raise ValueError("input_ids field not found in tokenized examples. Check tokenizer output.")
            
            # 优化：使用更高效的方式展平列表
            concatenated_examples = {}
            for k in available_fields:
                # 使用 chain 展平列表的列表（对于大规模数据，这是相对高效的方法）
                concatenated_examples[k] = list(chain(*examples[k]))
            
            # 使用 input_ids 的长度作为参考长度
            total_length = len(concatenated_examples['input_ids'])
            
            # 验证所有字段长度一致（tokenizer 应该保证这一点）
            for k in available_fields:
                if len(concatenated_examples[k]) != total_length:
                    raise ValueError(
                        f"Field {k} has length {len(concatenated_examples[k])}, "
                        f"but input_ids has length {total_length}. "
                        "All tokenizer fields must have the same length."
                    )
            
            # 我们丢掉最后一个小 block，不足 block_size 的部分会被丢弃
            total_length = (total_length // block_size) * block_size
            
            # 如果连接后的长度小于 block_size，返回空结果
            if total_length == 0:
                return {k: [] for k in available_fields + ['labels']}
            
            # 切分成 chunks（使用列表推导式，比循环更快）
            result = {}
            for k in available_fields:
                result[k] = [
                    concatenated_examples[k][i : i + block_size] 
                    for i in range(0, total_length, block_size)
                ]
            
            # 添加 labels（与 input_ids 相同，用于语言模型训练）
            result["labels"] = result["input_ids"].copy()
            
            return result

        # 根据disable_group_texts决定是否执行group_texts
        if disable_group_texts:
            print("[INFO] disable_group_texts=True, applying blocking (padding/truncation) to each sample...")
            # 导入blocking函数
            from lmflow.tokenization.hf_decoder_model import blocking
            
            def apply_blocking(examples):
                """对每个样本应用blocking（padding/truncation到block_size）"""
                tokenizer_fields = ['input_ids', 'attention_mask', 'special_tokens_mask']
                available_fields = [k for k in tokenizer_fields if k in examples]
                
                if 'input_ids' not in available_fields:
                    raise ValueError("input_ids field not found in tokenized examples.")
                
                # 对每个样本进行blocking
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                padding_side = tokenizer.padding_side
                
                # 创建labels（与input_ids相同）
                examples['labels'] = [ids.copy() for ids in examples['input_ids']]
                
                # 应用blocking
                num_examples = len(examples['input_ids'])
                for i in range(num_examples):
                    max_length = min(block_size, model_max_length)
                    pad_length = max_length - len(examples['input_ids'][i])
                    
                    if pad_length < 0:
                        # 截断过长的样本
                        for key in ['input_ids', 'attention_mask', 'labels']:
                            if key in examples:
                                examples[key][i] = examples[key][i][:max_length]
                    else:
                        # 对过短的样本进行padding
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
            
            # 应用blocking，batch_size=1确保每个样本独立处理
            lm_dataset = tokenized_ds.map(
                apply_blocking,
                batched=True,
                batch_size=1,  # 每个样本独立处理
                num_proc=num_proc,
                desc=f"Applying blocking to each sample (block_size={block_size})",
                load_from_cache_file=not getattr(data_args, 'overwrite_cache', False),
            )
        else:
            print("[INFO] disable_group_texts=False, grouping texts into blocks...")
            # 使用group_texts逻辑
            group_batch_size = getattr(data_args, 'group_texts_batch_size', 1000)
            
            # 优化：确保只保留必要的字段，移除其他不需要的字段
            # 注意：tokenized_ds 应该已经只包含 tokenizer 返回的字段了
            # 但为了安全，我们显式检查并移除不需要的字段
            columns_to_keep = ['input_ids', 'attention_mask', 'special_tokens_mask']
            columns_to_remove = [col for col in tokenized_ds.column_names if col not in columns_to_keep]
            
            if columns_to_remove:
                print(f"[INFO] Removing unnecessary columns before group_texts: {columns_to_remove}")
                tokenized_ds = tokenized_ds.remove_columns(columns_to_remove)
            
            lm_dataset = tokenized_ds.map(
                group_texts,
                batched=True,
                batch_size=group_batch_size,
                num_proc=num_proc,
                desc=f"Grouping texts in chunks of {block_size}",
                load_from_cache_file=not getattr(data_args, 'overwrite_cache', False),
            )

        train_dataset = lm_dataset
        print(f"[INFO] Final processed dataset has {len(train_dataset)} samples.")


    # 4. 【全新部分】定义 TrainingArguments 和 Trainer
    print("[INFO] Setting up transformers.TrainingArguments and Trainer...")

    # 将 pipeline_args 和其他参数手动映射到 TrainingArguments
    training_args = TrainingArguments(
        output_dir=pipeline_args.output_dir,
        do_train=True,
        num_train_epochs=pipeline_args.num_train_epochs,
        per_device_train_batch_size=pipeline_args.per_device_train_batch_size,
        gradient_accumulation_steps=pipeline_args.gradient_accumulation_steps,
        learning_rate=pipeline_args.learning_rate,
        lr_scheduler_type=pipeline_args.lr_scheduler_type,
        warmup_steps=getattr(pipeline_args, 'warmup_steps', 0), # 使用 warmup_steps
        bf16=pipeline_args.bf16,
        logging_steps=pipeline_args.logging_steps,
        save_steps=pipeline_args.save_steps,
        save_total_limit=pipeline_args.save_total_limit,
        gradient_checkpointing=pipeline_args.gradient_checkpointing,
        ddp_find_unused_parameters=getattr(pipeline_args, 'ddp_find_unused_parameters', None),
        report_to="wandb",
        run_name=getattr(pipeline_args, 'run_name', 'pretrain-gla'),
        dataloader_num_workers=getattr(pipeline_args, 'dataloader_num_workers', getattr(data_args, 'dataloader_num_workers', 16)),
        dataloader_drop_last=getattr(pipeline_args, 'dataloader_drop_last', True),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        max_steps=getattr(pipeline_args, 'max_steps', None),
    )

    # 定义数据整理器 Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 初始化 Trainer
    trainer = Trainer(
        model=gla_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 5. 开始训练
    print("[INFO] Starting training with native transformers.Trainer...")
    trainer.train()
    
    print("[INFO] Training finished!")


if __name__ == '__main__':
    main()
