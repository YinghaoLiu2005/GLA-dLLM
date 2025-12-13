#!/usr/bin/env python
# coding=utf-8
"""Automatically get correct model type.
"""


from .hf_decoder_model import HFDecoderModel
from .hf_encoder_decoder_model import HFEncoderDecoderModel
from .hf_text_regression_model import HFTextRegressionModel
from .gla_dream_model import GLADreamModel


class AutoModel:
    """AutoModel class to automatically select the correct model type."""

    @classmethod
    def get_model(self, model_args, *args, **kwargs):
        arch_type = model_args.arch_type
        if arch_type == "decoder_only":
            return HFDecoderModel(model_args, *args, **kwargs)
        elif arch_type == "text_regression":
            return HFTextRegressionModel(model_args, *args, **kwargs)
        elif arch_type == "encoder_decoder" or \
                arch_type == "vision_encoder_decoder":
            return HFEncoderDecoderModel(model_args, *args, **kwargs)
        elif arch_type == "gla_dream":
            return GLADreamModel(model_args, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"model architecture type \"{arch_type}\" is not supported"
            )
