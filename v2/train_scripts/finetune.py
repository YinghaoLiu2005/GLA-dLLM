#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import sys
import os
# sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser, AutoConfig, AutoModelForCausalLM
from BiDeltaDiff.models.initialization import load_partial_weights 
from BiDeltaDiff.models import BiDeltaDiffConfig,BiDeltaDiffForCausalLM
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline




def main():
    # Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    is_resuming = False
    # 检查 pipeline_args 或者 sys.argv
    if hasattr(pipeline_args, 'resume_from_checkpoint') and pipeline_args.resume_from_checkpoint:
        is_resuming = True
    # 双重保险：检查命令行参数
    for arg in sys.argv:
        if "--resume_from_checkpoint" in arg:
            is_resuming = True
            break
            
    if is_resuming:
        print("\n" + "!"*40)
        print(">>> 检测到恢复训练模式 (Resuming from Checkpoint)")
        print(">>> 步骤 A: 跳过 'load_partial_weights' (魔改初始化)")
        print(">>> 步骤 B: 保持 '冻结逻辑' 一致")
        print("!"*40 + "\n")


    # Add custom model path to sys.path to allow for dynamic import
    if model_args.custom_model_path is not None:
        # This allows transformers to find your custom modeling code
        sys.path.insert(0, model_args.custom_model_path)
        # Also add the root of the project to sys.path to find BiDeltaDiff
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, project_root)

        print(f"Loading custom model structure from {model_args.custom_model_path}")
        print(f"Initializing with weights from {model_args.model_name_or_path}")

        # 1. Load the custom model architecture with random weights
        # `trust_remote_code=True` is crucial for loading your custom python files.
        new_config = BiDeltaDiffConfig()
        backend_model = BiDeltaDiffForCausalLM(new_config)
        print("backend_model class:", backend_model.__class__)
        if not is_resuming:
        # 2. Load the state dict from the pretrained model
            print(f"Initializing with BASE weights from {model_args.model_name_or_path}")
            pretrained_state_dict = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).state_dict()

            # 3. Use your custom function to load partial weights
            backend_model = load_partial_weights(backend_model, pretrained_state_dict, verbose=True)
        else:
            print(">>> Skipping base weight loading (Waiting for Trainer to load Checkpoint...)")
        # 4. Wrap the loaded model with lmflow's model class
        model = AutoModel.get_model(model_args)
        model.backend_model = backend_model
    else:
        # Build model for fast-dLLM v2-1.5B (no mixed initialization here)
        model = AutoModel.get_model(model_args)

    # Freezing MLP
    target_model = model.backend_model
    
    frozen_keys = []
    for name, param in target_model.named_parameters():
        # 只要参数名包含 'mlp'，就将其冻结
        # 根据你之前的架构，名字通常是 layers.x.mlp.gate_proj.weight 等
        if "mlp" in name or "embed" in name:
            param.requires_grad = False
            frozen_keys.append(name)
        else:
            # 确保其他层（如 attn, embed, head）是可训练的
            param.requires_grad = True

    # --- 打印统计信息以验证 ---
    trainable_params = 0
    all_param = 0
    for _, param in target_model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f">>> 已冻结 MLP 层参数数量: {len(frozen_keys)}")
    print(f">>> 总参数量: {all_param / 1e9:.2f} B")
    print(f">>> 可训练参数量: {trainable_params / 1e9:.2f} B")
    print(f">>> 可训练比例: {100 * trainable_params / all_param:.2f}%")
    print("="*50 + "\n")


    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    data_args.file_pattern = "chat.jsonl"
    dataset = Dataset(data_args)

    # --- 新增：Overfitting 测试逻辑 ---
    # 仅当你想做 overfitting 测试时开启（可以通过环境变量或硬编码控制）
    # 这里为了方便直接硬编码，测试完记得注释掉
    DO_OVERFIT_TEST = True 
    if DO_OVERFIT_TEST:
        print("\n" + "!"*50)
        print(">>> [DEBUG] 开启 Overfitting 测试模式：仅使用前 100 条数据")
        
        backend_ds = dataset.backend_dataset
        
        # 处理 DatasetDict (通常包含 'train')
        if hasattr(backend_ds, "keys") and "train" in backend_ds:
            # 取前 10 条，如果不足 10 条则取全部
            num_samples = min(10, len(backend_ds["train"]))
            backend_ds["train"] = backend_ds["train"].select(range(num_samples))
            print(f">>> 已截取 'train' split 前 {num_samples} 条数据")
            
        # 处理单一 Dataset
        elif hasattr(backend_ds, "select"):
            num_samples = min(100, len(backend_ds))
            dataset.backend_dataset = backend_ds.select(range(num_samples))
            print(f">>> 已截取 Dataset 前 {num_samples} 条数据")
            
        print("!"*50 + "\n")
    # --------------------------------
    if (
        not is_resuming
        and data_args.dataset_num_shards > 1
    ):
        print(
            f">>> Using dataset shard "
            f"{data_args.dataset_shard_index}/{data_args.dataset_num_shards}"
        )

        backend_ds = dataset.backend_dataset

        # 情况 1：DatasetDict（最常见）
        if hasattr(backend_ds, "keys") and "train" in backend_ds:
            backend_ds["train"] = backend_ds["train"].shard(
                num_shards=data_args.dataset_num_shards,
                index=data_args.dataset_shard_index,
            )

        # 情况 2：单一 Dataset
        elif hasattr(backend_ds, "shard"):
            dataset.backend_dataset = backend_ds.shard(
                num_shards=data_args.dataset_num_shards,
                index=data_args.dataset_shard_index,
            )

        else:
            raise RuntimeError(
                f"Unknown backend_dataset type: {type(backend_ds)}"
            )
    else:
        print(">>> Using full dataset (no sharding or resuming)")

    # This part might need adjustment depending on your new model's config structure
    if hasattr(model.backend_model.config, 'bd_size'):
        data_args.bd_size = model.backend_model.config.bd_size
        data_args.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()