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

from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

# Import for mixed initialization
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'gla_dream'))
try:
    from gla_dream.modeling_gla_dream import GLADreamModel
    from gla_dream.configuration_gla_dream import GLADreamConfig
    GLA_DREAM_AVAILABLE = True
except ImportError:
    GLA_DREAM_AVAILABLE = False
    print("Warning: GLA Dream not available. Mixed initialization disabled.")


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

    # Mixed initialization from Qwen3 if specified
    if hasattr(model_args, 'use_mixed_init') and model_args.use_mixed_init and GLA_DREAM_AVAILABLE:
        print("Using mixed initialization from Qwen3...")
        # Create GLA Dream config compatible with Qwen3
        gla_config = GLADreamConfig(
            vocab_size=151936,  # Qwen3 vocab size
            hidden_size=1024,   # Qwen3-0.6B hidden size
            num_hidden_layers=24,  # Qwen3-0.6B layers
            num_attention_heads=16,  # Qwen3-0.6B heads
            intermediate_size=2816,  # Qwen3-0.6B intermediate size
            max_position_embeddings=32768,  # Qwen3 max position
            bd_size=getattr(data_args, 'bd_size', 32),
        )
        
        # Initialize GLA Dream model from Qwen3
        gla_model = GLADreamModel.from_qwen3_pretrained(
            qwen3_model_path=model_args.model_name_or_path,
            gla_config=gla_config
        )
        
        # Convert to AutoModel format for compatibility
        model = AutoModel.get_model(model_args)
        model.backend_model = gla_model
        
        print("Mixed initialization completed!")
    else:
        # Standard initialization
        model = AutoModel.get_model(model_args)

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    
    # Set bd_size from model config if available, otherwise use default
    if hasattr(model.backend_model.config, 'bd_size'):
        data_args.bd_size = model.backend_model.config.bd_size
    else:
        # Use default bd_size if not available in model config
        data_args.bd_size = getattr(data_args, 'bd_size', 32)
    
    data_args.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()
