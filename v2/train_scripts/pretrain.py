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


def main():
    # Parses arguments
    pipeline_name = "pretrainer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Build model: replace Dream backend with GLADream initialized from Fast-dLLM v2
    if GLA_DREAM_AVAILABLE:
        print("[Pretrain] Using GLADream initialized from Fast_dLLM_v2_1.5B...")

        # Load base config from the provided checkpoint to match shapes
        base_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )

        # Create GLADream config aligned with the base model
        gla_config = GLADreamConfig(
            vocab_size=getattr(base_config, 'vocab_size', 151936),
            hidden_size=getattr(base_config, 'hidden_size', 1024),
            num_hidden_layers=getattr(base_config, 'num_hidden_layers', 24),
            num_attention_heads=getattr(base_config, 'num_attention_heads', 16),
            intermediate_size=getattr(base_config, 'intermediate_size', 2816),
            max_position_embeddings=getattr(base_config, 'max_position_embeddings', 32768),
            bd_size=getattr(data_args, 'bd_size', getattr(base_config, 'bd_size', 32)),
        )

        # Initialize GLADream by reusing compatible weights from the base checkpoint
        # Incompatible/new modules are left randomly initialized by GLADream
        gla_model = GLADreamModel.from_fast_dllm_pretrained(
            fast_dllm_model_path=model_args.model_name_or_path,
            gla_config=gla_config,
            trust_remote_code=model_args.trust_remote_code,
        )

        # Build wrapper model to keep tokenizer / lmflow integration
        model = AutoModel.get_model(model_args)
        model.backend_model = gla_model

        # Freeze inherited weights if requested (only train randomly initialized modules)
        if model_args.freeze_inherited_weights:
            print("[Pretrain] Freezing inherited weights from Fast-dLLM, only training new/randomly initialized modules...")
            _freeze_inherited_weights(gla_model, model_args.model_name_or_path, model_args.trust_remote_code)
            print("[Pretrain] Weight freezing completed!")

        print("[Pretrain] GLADream mixed initialization completed!")
    else:
        # Fallback: use standard model construction (will use original Dream backend)
        model = AutoModel.get_model(model_args)

    # Initialization
    pretrainer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    data_args.dataset_type = 'parquet'
    dataset = Dataset(data_args)

    # Ensure bd_size and mask_id are aligned with the backend
    if hasattr(model.backend_model.config, 'bd_size'):
        data_args.bd_size = model.backend_model.config.bd_size
    else:
        data_args.bd_size = getattr(data_args, 'bd_size', 32)

    data_args.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # Continuous pretraining
    # Finetuner pipeline exposes `tune`, not `train`/`pretrain`.
    if hasattr(pretrainer, 'tune'):
        _ = pretrainer.tune(model=model, dataset=dataset)
    elif hasattr(pretrainer, 'train'):
        _ = pretrainer.train(model=model, dataset=dataset)
    elif hasattr(pretrainer, 'pretrain'):
        _ = pretrainer.pretrain(model=model, dataset=dataset)
    else:
        raise AttributeError("Pretrainer pipeline does not define `tune`, `train` or `pretrain`.")


if __name__ == '__main__':
    main()
