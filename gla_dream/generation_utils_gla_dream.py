# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    """Apply top-p (nucleus) filtering"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    """Apply top-k filtering"""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    """Sample tokens from logits"""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class GLADreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class GLADreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # Generation specific params
        self.threshold: float = kwargs.pop("threshold", 0.9)
        self.steps: int = kwargs.pop("steps", 512)
        
        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Additional attributes
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


class GLADreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _prepare_generation_config(
        self, generation_config: Optional[GLADreamGenerationConfig], **kwargs: Dict
    ) -> GLADreamGenerationConfig:
        """Prepares the base generation config"""
        using_model_generation_config = False
        if generation_config is None:
            generation_config = GLADreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)

            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: GLADreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Prepares the special tokens for generation"""
        def _tensor_or_none(token, device=None):
            if token is None:
                return token
            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GLADreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[GLADreamModelOutput, torch.LongTensor]:
        """
        Block-wise recursive generation using GLA-Dream architecture.
        
        The generation process:
        1. Process prompt through parallel mode to get initial inter-block state S_prompt
        2. For each new block i:
           - Get historical state: S_{i-1} (initially S_prompt)
           - Initialize current block with [MASK] tokens
           - Iterative denoising:
             * Inject S_{i-1} into current_block
             * Process through bidirectional GLA (parallel for entire block)
             * Decode high-confidence tokens
             * Update current_block
             * Repeat until block is complete
           - Update global state via inter-block recurrent unit
        """
        # Prepare generation config
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # Get generation parameters
        threshold = kwargs.get("threshold", generation_config.threshold)
        block_length = self.config.block_size
        steps = kwargs.get("steps", generation_config.steps)
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_history = generation_config.output_history

        histories = [] if (return_dict_in_generate and output_history) else None

        # Prepare input: pad to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, 0.001, steps_per_block + 1, device=x.device)

        # Initialize past_key_values (inter-block states)
        past_key_values = None

        # Process prompt and get initial inter-block state
        # This processes the prompt through all layers to get initial S_prompt
        current_block_start = input_ids.shape[1]
        
        # Generate each block sequentially
        for num_block in range(num_blocks):
            current_block_start_idx = input_ids.shape[1] + num_block * block_length
            current_block_end_idx = current_block_start_idx + block_length

            # Extract the current block (initialize with [MASK] tokens)
            current_block_tokens = x[:, current_block_start_idx:current_block_end_idx]
            mask_index = (current_block_tokens == mask_token_id)

            # Iterative denoising loop for this block
            iteration = 0
            while mask_index.sum() > 0 and iteration < steps_per_block:
                # Process the block through the model
                # The model will handle inter-block state injection internally
                model_output = self(input_ids=x, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output.past_key_values
                logits = model_output.logits
                
                # Apply token shift for prediction
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Sample tokens
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                
                # Update only tokens above confidence threshold
                mask_index = (x[:, current_block_start_idx:current_block_end_idx] == mask_token_id)
                if mask_index.sum() > 0:
                    mask_logits = logits[:, current_block_start_idx:current_block_end_idx][mask_index]
                    mask_confidence, mask_x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    # Update tokens with confidence above threshold
                    high_conf_mask = mask_confidence > threshold
                    if high_conf_mask.sum() > 0:
                        update_positions = mask_index.nonzero(as_tuple=False)
                        for pos in update_positions:
                            if high_conf_mask[pos[1].item()]:
                                x[0, current_block_start_idx + pos[1].item()] = mask_x0[pos[1].item()]

                iteration += 1

                # Check if we're done with this block
                mask_index = (x[:, current_block_start_idx:current_block_end_idx] == mask_token_id)
                if mask_index.sum() == 0:
                    break

            # After completing this block, update past_key_values with the new inter-block state
            # This will be handled by the model's internal state management

        if return_dict_in_generate:
            return GLADreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

