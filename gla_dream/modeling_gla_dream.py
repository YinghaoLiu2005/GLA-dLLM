# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT and Qwen implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT and Qwen used by the Meta AI and Qwen team that trained the model.
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
"""PyTorch GLADream model."""

import math
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MaskedLMOutput,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
)
from transformers import PretrainedConfig
from .configuration_gla_dream import GLADreamConfig
from .generation_utils_gla_dream import GLADreamGenerationMixin, GLADreamGenerationConfig

try:
    from fla.layers.gla import GatedLinearAttention
except ImportError:
    logging.warning_once("fla.layers.gla not available, falling back to fallback implementation")
    GatedLinearAttention = None

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "GLADream-7B"
_CONFIG_FOR_DOC = "GLADreamConfig"

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->GLADream
class GLADreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GLADreamRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->GLADream
class GLADreamMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class BiDirectionalGLABlock(nn.Module):
    """
    Intra-block bidirectional GLA unit.
    Processes a block bidirectionally and fuses the results via gating mechanism.
    """
    def __init__(self, config: GLADreamConfig, layer_idx: int):
        super().__init__()
        self.config = config
        
        if GatedLinearAttention is None:
            raise ImportError("GatedLinearAttention from fla.layers.gla is required")
        
        # Forward and backward GLA layers
        self.forward_gla = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
        )


        # Gating mechanism for fusion
        self.gate_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            fused_h: (batch_size, seq_len, hidden_size)
            h_forward: (batch_size, seq_len, hidden_size)
            h_backward: (batch_size, seq_len, hidden_size)
        """
        # Forward scan
        h_forward = self.forward_gla(hidden_states)
        if isinstance(h_forward, tuple):
            h_forward = h_forward[0]
        
        # Backward scan (flip, process, flip back)
        h_backward = self.forward_gla(torch.flip(hidden_states, [1]))
        if isinstance(h_backward, tuple):
            h_backward = h_backward[0]
        h_backward = torch.flip(h_backward, [1])

        # Fusion via gating
        gate_input = torch.cat([h_forward, h_backward], dim=-1)
        g = torch.sigmoid(self.gate_proj(gate_input))
        
        fused_h = g * h_forward + (1 - g) * h_backward
        return fused_h, h_forward, h_backward

class GLADreamDecoderLayer(nn.Module):
    """
    Complete decoder layer with:
    1. Inter-block recurrent unit (unidirectional GLA)
    2. Intra-block bidirectional unit (BiDirectionalGLABlock)
    3. FFN
    """
    def __init__(self, config: GLADreamConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Inter-block recurrent unit (unidirectional GLA for context propagation)
        if GatedLinearAttention is None:
            raise ImportError("GatedLinearAttention from fla.layers.gla is required")
        
        # Note: Inter-block GLA is currently not used as it would require different processing
        # It's kept here for future implementation of inter-block recurrent processing
        # self.inter_block_gla = GatedLinearAttention(...)
        
        # Intra-block bidirectional unit
        self.intra_block_gla = BiDirectionalGLABlock(config, layer_idx)
        
        # Layer norms
        self.input_layernorm = GLADreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GLADreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_inter_layernorm = GLADreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # FFN
        self.mlp = GLADreamMLP(config)

        # Gating for inter-block state fusion (independent from intra-block gate)
        if config.share_inter_intra_gate_weights:
            self.inter_block_gate_proj = self.intra_block_gla.gate_proj
        else:
            self.inter_block_gate_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inter_block_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.Tensor]:
        """
        Args:
            hidden_states: Current block hidden states (batch_size, block_size, hidden_size)
            inter_block_state: Previous block's output state (batch_size, hidden_size)
            
        Returns:
            hidden_states: Processed block hidden states
            next_inter_block_state: New state for next block
        """
        residual = hidden_states
        
        # 1. Inter-block state injection
        if inter_block_state is not None:
            if self.config.inter_block_injection_mode == "global":
                # Inject to every token in the block
                hidden_states = hidden_states + inter_block_state.unsqueeze(1)
            elif self.config.inter_block_injection_mode == "boundary":
                # Inject only to boundaries
                hidden_states[:, 0] += inter_block_state
                hidden_states[:, -1] += inter_block_state
        
        # 2. Intra-block bidirectional processing
        hidden_states_norm = self.input_layernorm(hidden_states)
        fused_h, h_forward, h_backward = self.intra_block_gla(hidden_states_norm)
        hidden_states = residual + fused_h
        
        # 3. FFN after intra-block processing
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states_norm)
        hidden_states = residual + hidden_states_mlp
        
        # 4. Compute new inter-block state via gating
        h_last_forward = h_forward[:, -1]  # Last token's forward state
        h_first_backward = h_backward[:, 0]  # First token's backward state
        
        gate_input = torch.cat([h_last_forward, h_first_backward], dim=-1)
        gate = torch.sigmoid(self.inter_block_gate_proj(gate_input))
        next_inter_block_state = gate * h_last_forward + (1 - gate) * h_first_backward
        # 5.Normalize the state before passing it to the next block for stability
        next_inter_block_state = self.post_inter_layernorm(next_inter_block_state)
        
        return hidden_states, next_inter_block_state

class GLADreamPreTrainedModel(PreTrainedModel):
    config_class = GLADreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GLADreamDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        # Override generation config
        resume_download = kwargs.get("resume_download", None)
        proxies = kwargs.get("proxies", None)
        subfolder = kwargs.get("subfolder", "")
        from_auto_class = kwargs.get("_from_auto", False)
        from_pipeline = kwargs.get("_from_pipeline", None)
        _model.generation_config = GLADreamGenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
        )
        return _model

class GLADreamBaseModel(GLADreamPreTrainedModel):
    def __init__(self, config: GLADreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Build layers: allow mixing GLA-based layers and Dream decoder layers.
        # If config.gla_layer_indices is None or empty, default to all GLA layers
        self.layers = nn.ModuleList()
        gla_indices = getattr(config, "gla_layer_indices", None)
        # Normalize gla_indices to a set for fast lookup
        if gla_indices is None:
            gla_set = None
        else:
            gla_set = set(gla_indices) if isinstance(gla_indices, (list, tuple, set)) else {int(gla_indices)}

        for layer_idx in range(config.num_hidden_layers):
            use_gla = gla_set is None or (layer_idx in gla_set)
            if use_gla:
                # Use GLA-based layer
                self.layers.append(GLADreamDecoderLayer(config, layer_idx))
            else:
                # Try to fall back to Dream's decoder layer for standard attention
                try:
                    # Import here to avoid circular import issues when module not available
                    from dream.model.modeling_dream import DreamDecoderLayer

                    self.layers.append(DreamDecoderLayer(config, layer_idx))
                except Exception:
                    # If Dream's implementation isn't importable, fall back to GLA layer
                    logging.get_logger(__name__).warning_once(
                        f"DreamDecoderLayer not available for layer {layer_idx}; falling back to GLA layer."
                    )
                    self.layers.append(GLADreamDecoderLayer(config, layer_idx))
        self.norm = GLADreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,  # Holds inter-block states
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds

        collected_hidden_states: Optional[List[List[torch.Tensor]]] = None
        if output_hidden_states:
            num_layers = len(self.layers)
            collected_hidden_states = [[] for _ in range(num_layers + 1)]
            collected_hidden_states[0].append(hidden_states)

        # During training, process the whole sequence as blocks
        # During inference, process one block at a time
        num_blocks = hidden_states.shape[1] // self.block_size
        if num_blocks > 1 and not self.training:
            logger.warning_once("Inference should be done one block at a time for efficiency.")
        
        # Reshape for block processing
        blocks = hidden_states.view(hidden_states.shape[0], num_blocks, self.block_size, hidden_states.shape[-1])

        output_blocks = []
        current_states_per_layer = past_key_values

        for i in range(num_blocks):
            current_block = blocks[:, i, :, :]
            
            block_output, new_states_per_layer, block_hidden_trace = self.process_block(
                current_block, 
                past_inter_block_states=current_states_per_layer,
                collect_hidden_states=output_hidden_states,
            )
            output_blocks.append(block_output)
            current_states_per_layer = new_states_per_layer
            if output_hidden_states and collected_hidden_states is not None:
                for layer_idx, layer_hidden in enumerate(block_hidden_trace, start=1):
                    collected_hidden_states[layer_idx].append(layer_hidden)
        
        hidden_states = torch.cat(output_blocks, dim=1)
        hidden_states = self.norm(hidden_states)

        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_hidden_states and collected_hidden_states is not None:
            all_hidden_states = tuple(torch.cat(layer_states, dim=1) if len(layer_states) > 1 else layer_states[0]
                                      for layer_states in collected_hidden_states)
        
        present_states = current_states_per_layer if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, present_states, all_hidden_states, None] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_states,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def process_block(
        self, 
        block_hidden_states: torch.Tensor,
        past_inter_block_states: Optional[Tuple[torch.Tensor]] = None,
        collect_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Process a single block and return output + new inter-block states"""
        
        new_states_across_layers = []
        layer_hidden_trace: Optional[List[torch.Tensor]] = [] if collect_hidden_states else None

        for i, layer in enumerate(self.layers):
            layer_past_state = past_inter_block_states[i] if past_inter_block_states is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)[0]
                    return custom_forward
                
                block_hidden_states_temp = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    block_hidden_states,
                    layer_past_state,
                    use_reentrant=False
                )
                # Recompute to get the state without checkpointing
                with torch.no_grad():
                    _, layer_present_state = layer(block_hidden_states, layer_past_state)
                block_hidden_states = block_hidden_states_temp
            else:
                block_hidden_states, layer_present_state = layer(
                    block_hidden_states, 
                    inter_block_state=layer_past_state
                )

            new_states_across_layers.append(layer_present_state)
            if collect_hidden_states and layer_hidden_trace is not None:
                layer_hidden_trace.append(block_hidden_states)
        
        return block_hidden_states, tuple(new_states_across_layers), layer_hidden_trace

class GLADreamModel(GLADreamGenerationMixin, GLADreamPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GLADreamBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction='none')        

        # Initialize weights and apply final processing
        self.post_init()

    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向传播函数，支持Dream mask策略和Fast-dLLM v2 complementary mask策略
        
        Args:
            input_ids: Token IDs. If complementary mask is used, shape is (2 * batch_size, seq_len)
            labels: Optional labels for loss calculation
        """
        # Remove incompatible parameters
        kwargs.pop('is_causal', None)
        kwargs.pop('attention_mask', None)
        kwargs.pop('num_items_in_batch', None)  # Remove training-specific parameter

        # 1. Get hidden states from model
        outputs = self.model(input_ids=input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state

        # 2. Implement token shift for prediction
        # Shift hidden states to the right for predicting i using hidden state i-1
        shifted_hidden_states = F.pad(hidden_states[:, :-1, :], (0, 0, 1, 0))

        # 3. Compute logits
        logits = self.lm_head(shifted_hidden_states)

        loss = None
        if labels is not None:
            # 4. Compute per-token loss (CrossEntropyLoss(ignore_index=-100, reduction='none'))
            loss_per_token = self.loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            valid_mask = (labels.view(-1) != -100).to(loss_per_token.dtype)
            valid_tokens = valid_mask.sum().clamp_min(1)
            loss = (loss_per_token * valid_mask).sum() / valid_tokens

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_qwen3_pretrained(
        cls,
        qwen3_model_path: str,
        gla_config: "GLADreamConfig",
        **kwargs
    ):
        """
        从Qwen3预训练模型初始化GLA Dream模型
        
        Args:
            qwen3_model_path: Qwen3模型路径
            gla_config: GLA Dream配置
            **kwargs: 其他参数
        """
        # 1. 加载Qwen3模型
        from transformers import AutoModel
        qwen3_model = AutoModel.from_pretrained(qwen3_model_path, **kwargs)
        qwen3_state_dict = qwen3_model.state_dict()
        
        # 2. 创建GLA Dream模型
        gla_model = cls(gla_config)
        gla_state_dict = gla_model.state_dict()
        
        # 3. 定义可复用的组件映射
        compatible_mappings = {
            # 词嵌入层
            'model.embed_tokens.weight': 'model.embed_tokens.weight',
            # MLP层
            'model.layers.{}.mlp.gate_proj.weight': 'model.layers.{}.mlp.gate_proj.weight',
            'model.layers.{}.mlp.up_proj.weight': 'model.layers.{}.mlp.up_proj.weight', 
            'model.layers.{}.mlp.down_proj.weight': 'model.layers.{}.mlp.down_proj.weight',
            # LayerNorm
            'model.layers.{}.input_layernorm.weight': 'model.layers.{}.input_layernorm.weight',
            'model.layers.{}.post_attention_layernorm.weight': 'model.layers.{}.post_attention_layernorm.weight',
            'model.norm.weight': 'model.norm.weight',
            # 输出头
            'lm_head.weight': 'lm_head.weight',
        }
        
        # 4. 复制兼容的权重
        for qwen_key, gla_key in compatible_mappings.items():
            if '{' in qwen_key:  # 处理层索引
                for layer_idx in range(gla_config.num_hidden_layers):
                    qwen_layer_key = qwen_key.format(layer_idx)
                    gla_layer_key = gla_key.format(layer_idx)
                    if qwen_layer_key in qwen3_state_dict and gla_layer_key in gla_state_dict:
                        gla_state_dict[gla_layer_key] = qwen3_state_dict[qwen_layer_key].clone()
            else:
                if qwen_key in qwen3_state_dict and gla_key in gla_state_dict:
                    gla_state_dict[gla_key] = qwen3_state_dict[qwen_key].clone()
        
        # 5. 加载更新后的状态字典
        gla_model.load_state_dict(gla_state_dict)
        
        return gla_model

    @classmethod
    def from_fast_dllm_pretrained(
        cls,
        fast_dllm_model_path: str,
        gla_config: "GLADreamConfig",
        **kwargs
    ):
        """
        使用 Fast_dLLM_v2_1.5B 预训练权重初始化 GLA Dream 模型。

        可复用模块直接拷贝，其余新模块随机初始化。
        """
        # 1. 加载 Fast-dLLM 模型（依赖其 config 中的 auto_map）
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(fast_dllm_model_path, **kwargs)
        base_state_dict = base_model.state_dict()

        # 2. 创建 GLA Dream 模型
        gla_model = cls(gla_config)
        gla_state_dict = gla_model.state_dict()

        # 3. 定义可复用的组件映射（仅拷贝形状/语义对齐的模块）
        compatible_mappings = {
            # 词嵌入层
            'model.embed_tokens.weight': 'model.embed_tokens.weight',
            'embed_tokens.weight': 'model.embed_tokens.weight',
            # MLP层
            'model.layers.{}.mlp.gate_proj.weight': 'model.layers.{}.mlp.gate_proj.weight',
            'model.layers.{}.mlp.up_proj.weight': 'model.layers.{}.mlp.up_proj.weight',
            'model.layers.{}.mlp.down_proj.weight': 'model.layers.{}.mlp.down_proj.weight',
            'layers.{}.mlp.gate_proj.weight': 'model.layers.{}.mlp.gate_proj.weight',
            'layers.{}.mlp.up_proj.weight': 'model.layers.{}.mlp.up_proj.weight',
            'layers.{}.mlp.down_proj.weight': 'model.layers.{}.mlp.down_proj.weight',
            # LayerNorm（post_attention_layernorm 映射到 GLA 的 post_attention_layernorm）
            'model.layers.{}.input_layernorm.weight': 'model.layers.{}.input_layernorm.weight',
            'model.layers.{}.post_attention_layernorm.weight': 'model.layers.{}.post_attention_layernorm.weight',
            'layers.{}.input_layernorm.weight': 'model.layers.{}.input_layernorm.weight',
            'layers.{}.post_attention_layernorm.weight': 'model.layers.{}.post_attention_layernorm.weight',
            # 输出层 Norm 与 lm_head
            'model.norm.weight': 'model.norm.weight',
            'norm.weight': 'model.norm.weight',
            'lm_head.weight': 'lm_head.weight',
        }

        # 4. 复制兼容的权重（逐层处理）
        for base_key, gla_key in compatible_mappings.items():
            if '{' in base_key:
                for layer_idx in range(gla_config.num_hidden_layers):
                    b_key = base_key.format(layer_idx)
                    g_key = gla_key.format(layer_idx)
                    if b_key in base_state_dict and g_key in gla_state_dict and base_state_dict[b_key].shape == gla_state_dict[g_key].shape:
                        gla_state_dict[g_key] = base_state_dict[b_key].clone()
                        print(f"{g_key} successfully initialization from fast-dllm_v2 (source: {b_key})")
                    else:
                        # if no matching source or shape mismatch, we keep random init for this target key
                        if g_key in gla_state_dict:
                            print(f"{g_key} initialization from scratch")
            else:
                # try copy from base_key -> gla_key when both exist and shapes match
                if base_key in base_state_dict and gla_key in gla_state_dict and base_state_dict[base_key].shape == gla_state_dict[gla_key].shape:
                    gla_state_dict[gla_key] = base_state_dict[base_key].clone()
                    print(f"{gla_key} successfully initialization from fast-dllm_v2 (source: {base_key})")
                # fallback: maybe base_key itself is present in gla_state_dict (source has no "model." prefix)
                elif base_key in base_state_dict and base_key in gla_state_dict and base_state_dict[base_key].shape == gla_state_dict[base_key].shape:
                    gla_state_dict[base_key] = base_state_dict[base_key].clone()
                    print(f"{base_key} successfully initialization from fast-dllm_v2 (source: {base_key})")
                else:
                    # nothing matched; if target exists, note it will be randomly initialized
                    if gla_key in gla_state_dict:
                        print(f"{gla_key} initialization from scratch")

        # 5. 加载更新后的状态字典
        gla_model.load_state_dict(gla_state_dict, strict=False)

        return gla_model

    def generate(self, *args, **kwargs):
        """Forward to diffusion-based generation"""
        return self.diffusion_generate(*args, **kwargs)

from transformers import AutoModel

# Register the model
AutoModel.register(GLADreamConfig, GLADreamModel)

# For compatibility with wrappers expecting a CausalLM class name
GLADreamForCausalLM = GLADreamModel
