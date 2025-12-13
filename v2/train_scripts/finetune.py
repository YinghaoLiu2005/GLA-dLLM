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
from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

# BiDeltaDiff 权重部分加载工具
from BiDeltaDiff.models.initialization import load_partial_weights


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

    # Build model for fast-dLLM v2-1.5B
    model = AutoModel.get_model(model_args)

    # 如果提供了旧模型权重路径，则进行部分权重加载
    # 约定使用 HF-style 的 "model_name_or_path" 字段传入旧权重目录
    old_model_path = getattr(model_args, "model_name_or_path", None)
    if old_model_path is not None and os.path.isdir(old_model_path):
        try:
            from transformers import AutoModelForCausalLM

            print(f"[Init] 从 {old_model_path} 加载旧模型权重用于部分初始化...")
            old_model = AutoModelForCausalLM.from_pretrained(
                old_model_path,
                trust_remote_code=True,
            )
            old_state_dict = old_model.state_dict()
            model.backend_model = load_partial_weights(model.backend_model, old_state_dict)
            print("[Init] 部分权重加载完成。")
        except Exception as e:
            print(f"[Init] 警告：部分权重加载失败，改为随机初始化。错误信息: {e}")

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    
    data_args.bd_size = model.backend_model.config.bd_size
    data_args.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()