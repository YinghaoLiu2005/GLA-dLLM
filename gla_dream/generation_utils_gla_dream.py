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
        self.max_length = kwargs.pop("max_length", 512)
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
    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )        

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config
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
        修正后的分块生成函数，严格遵循模型的逐块循环状态传递机制。
        """
        # 1. 准备配置 (不变)
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        self._prepare_special_tokens(generation_config, device=device)
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config, has_default_max_length=has_default_max_length, input_ids_length=input_ids_length
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 2. 获取生成参数 (不变)
        threshold = kwargs.get("threshold", generation_config.threshold)
        block_length = self.config.block_size
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_history = generation_config.output_history
        histories = [] if (return_dict_in_generate and output_history) else None

        # 3. 准备输入序列 (不变)
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks_to_generate = gen_length // block_length
        assert steps % num_blocks_to_generate == 0, f"steps ({steps}) must be divisible by num_blocks_to_generate ({num_blocks_to_generate})"
        steps_per_block = steps // num_blocks_to_generate

        # 4. --- 核心逻辑重构 ---

        # 初始化跨块循环状态 (Inter-block State)
        # 它的结构是每层一个状态张量，所以是一个元组
        past_states_per_layer = None

        # 步骤A: 处理输入的Prompt，获得初始的循环状态
        # 这是至关重要的一步，为真正的生成提供起始上下文
        if input_ids.shape[1] > 0:
            prompt_embeds = self.model.embed_tokens(input_ids)
            num_prompt_blocks = prompt_embeds.shape[1] // block_length
            prompt_blocks = prompt_embeds.view(prompt_embeds.shape[0], num_prompt_blocks, self.config.block_size, prompt_embeds.shape[-1])
            
            for i in range(num_prompt_blocks):
                # 依次处理每个prompt块，并更新循环状态
                _, past_states_per_layer = self.model.process_block(
                    prompt_blocks[:, i, :, :],
                    past_inter_block_states=past_states_per_layer
                )

        # 步骤B: 逐个生成新的块
        for block_idx in range(num_blocks_to_generate):
            current_block_start_idx = input_ids.shape[1] + block_idx * block_length
            current_block_end_idx = current_block_start_idx + block_length

            # 初始化当前待生成的块，全部为 [MASK]
            current_block_tokens = x[:, current_block_start_idx:current_block_end_idx]

            # 步骤C: 在当前块内进行迭代去噪
            for iteration in range(steps_per_block):
                mask_index = (current_block_tokens == mask_token_id)
                if mask_index.sum() == 0:
                    break  # 如果已全部去噪，提前结束

                # 准备当前块的输入
                block_embeds = self.model.embed_tokens(current_block_tokens)

                # 调用 process_block，这是与模型架构匹配的关键！
                # 我们传入当前块的嵌入和上一个块传来的状态
                block_hidden_states, _ = self.model.process_block(
                    block_embeds,
                    past_inter_block_states=past_states_per_layer
                )
                
                # 手动模拟顶层模型的token shift和lm_head来获取logits
                shifted_hidden_states = F.pad(block_hidden_states[:, :-1, :], (0, 0, 1, 0))
                logits = self.lm_head(shifted_hidden_states)

                # 采样和更新逻辑 (与您原版类似)
                mask_logits = logits[mask_index]
                mask_confidence, mask_x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                
                high_conf_mask = mask_confidence > threshold
                if high_conf_mask.sum() > 0:
                    # 获取需要更新的位置 (在当前块内的相对位置)
                    update_indices_in_block = mask_index.nonzero(as_tuple=True)
                    confident_updates = high_conf_mask.nonzero(as_tuple=True)[0]
                    rows_to_update = update_indices_in_block[0][confident_updates]
                    cols_to_update = update_indices_in_block[1][confident_updates]
                    values_to_update = mask_x0[confident_updates]
                    
                    # 更新当前块的tokens
                    current_block_tokens[rows_to_update, cols_to_update] = values_to_update
            
            # 步骤D: 当前块去噪完成，将其更新回完整序列 x
            x[:, current_block_start_idx:current_block_end_idx] = current_block_tokens

            # 步骤E: 为下一个块的生成，计算并更新循环状态
            # 使用刚刚完成去噪的、干净的块来计算传递给下一个块的最终状态
            final_block_embeds = self.model.embed_tokens(current_block_tokens)
            _, past_states_per_layer = self.model.process_block(
                final_block_embeds,
                past_inter_block_states=past_states_per_layer
            )

        if return_dict_in_generate:
            return GLADreamModelOutput(sequences=x, history=histories)
        else:
            return x