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
        # 2. Load the state dict from the pretrained model
        pretrained_state_dict = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).state_dict()

        # 3. Use your custom function to load partial weights
        backend_model = load_partial_weights(backend_model, pretrained_state_dict, verbose=True)
        
        # 4. Wrap the loaded model with lmflow's model class
        model = AutoModel.get_model(model_args)
        model.backend_model = backend_model
    else:
        # Build model for fast-dLLM v2-1.5B (no mixed initialization here)
        model = AutoModel.get_model(model_args)

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    
    # This part might need adjustment depending on your new model's config structure
    if hasattr(model.backend_model.config, 'bd_size'):
        data_args.bd_size = model.backend_model.config.bd_size
        data_args.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()