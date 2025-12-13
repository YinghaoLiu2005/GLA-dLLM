from typing import Callable, Optional, Tuple, Union,List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from .configuration import BiDeltaDiffConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange, repeat

from fla.layers.kda import KimiDeltaAttention
from fla.modules import RMSNorm
from fla.modules import GatedMLP
from fla.modules.l2warp import l2_warp

logger = logging.get_logger(__name__)

@dataclass
class CausalLMOutputWithPastAndBlockCache(CausalLMOutputWithPast):
    block_past_key_values: Optional[Cache] = None

@dataclass
class BaseModelOutputWithPastAndBlockCache(BaseModelOutputWithPast):
    block_past_key_values: Optional[Cache] = None


class BiDeltaDiffAttention(nn.Module):
    """
    双向 Kimi Delta Attention 模块
    包含两个 KimiDeltaAttention 实例：一个处理正向，一个处理反向
    """
    def __init__(self, config: BiDeltaDiffConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_bidirectional = config.is_bidirectional
        self.out_project = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # 正向流 (Forward Stream)
        self.fwd_attn = KimiDeltaAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_key_value_heads,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            chunk_size=config.chunk_size, # 关键参数
            layer_idx=layer_idx,
            # 这里的 norm_eps 传给 KDA 内部的 norm
            norm_eps=config.rms_norm_eps 
        )

        # 反向流 (Backward Stream) - 只有开启双向才初始化
        if self.is_bidirectional:
            self.bwd_attn = KimiDeltaAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_key_value_heads,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                chunk_size=config.chunk_size,
                layer_idx=layer_idx,
                norm_eps=config.rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        # 1. Forward Pass
        # KDA 的 forward 返回 (hidden_states, attentions, past_key_values)
        out_fwd, attns_fwd, past_key_values = self.fwd_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        if not self.is_bidirectional:
            return out_fwd, attns_fwd, past_key_values
        
            
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()    
        # 2. Backward Pass (if bidirectional)
        # Flip input: [Batch, Seq, Dim] -> flip along dim 1
        hidden_states_rev = torch.flip(hidden_states, dims=[1]).contiguous()
        
        # 注意：反向流不需要 cache，因为扩散通常是一次性并行计算
        # 且 mask 也需要翻转 (如果有的话，通常 DeltaNet 不需要 mask)
        out_bwd, attns_bwd, _ = self.bwd_attn(
            hidden_states=hidden_states_rev,
            attention_mask=None, # 线性注意力通常处理 mask 的方式不同，这里假设是 causal 或 full
            past_key_values=None,
            use_cache=False, 
            output_attentions=output_attentions,
            **kwargs
        )

        # Flip output back
        out_bwd = torch.flip(out_bwd, dims=[1])

        # 3. Fusion (Sum)
        output = self.out_project(0.5 * torch.cat([out_fwd, out_bwd], dim=-1))  # Concatenate along feature dimension
        
        # Merge attentions if requested (optional debugging)
        attentions = None
        if output_attentions:
            attentions = (attns_fwd, attns_bwd)

        return output, attentions, past_key_values


class BiDeltaDiffBlock(nn.Module):
    def __init__(self, config: BiDeltaDiffConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        NormClass = RMSNorm if config.fuse_norm else nn.RMSNorm
        
        self.attn_norm = NormClass(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BiDeltaDiffAttention(config, layer_idx)
        
        self.mlp_norm = NormClass(config.hidden_size, eps=config.rms_norm_eps)
        act_fn_name=config.hidden_act
        if act_fn_name =="silu":
            act_fn_name ="swish"
        # 使用 FLA 优化的 GatedMLP (SwiGLU)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=act_fn_name,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # Pre-Norm Architecture
        hidden_states = self.attn_norm(hidden_states)
        
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # Residual 1
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual 2
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values


class BiDeltaDiffPreTrainedModel(PreTrainedModel):
    config_class = BiDeltaDiffConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiDeltaDiffBlock"]
    _skip_keys_device_placement = "past_key_values"

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
        
        # KDA 特有的初始化逻辑 (参考 fla 源码)
        if isinstance(module, KimiDeltaAttention):
            # 这里可以保留 KDA 的特殊初始化，或者简单用正态分布
            pass


class BiDeltaDiffModel(BiDeltaDiffPreTrainedModel):
    def __init__(self, config: BiDeltaDiffConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bd_size = config.bd_size # Block diffusion size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BiDeltaDiffBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        NormClass = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.norm = NormClass(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        

        next_decoder_cache=[] if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past = past_key_values[idx] if past_key_values is not None else None

            hidden_states, attentions, new_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                past_key_values=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )

            if use_cache:
                next_decoder_cache.append(new_states)

            if output_attentions:
                all_attentions += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class BiDeltaDiffForCausalLM(BiDeltaDiffPreTrainedModel, GenerationMixin):
    """
    外层 Wrapper:包含了 Fast_dLLM 的核心扩散训练逻辑
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BiDeltaDiffModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        mask_id: Optional[int] = 151665,
        **kwargs
    ) -> CausalLMOutputWithPastAndBlockCache:
        """
        这里是 Fast-dLLM 的核心逻辑：
        它在 forward 内部进行 Mask 采样（加噪），并构造 Complementary Mask。
        """
        
        # --- Fast-dLLM Diffusion Logic Start ---
        if self.training:
            original_labels = labels.clone()
            original_input_ids = input_ids.clone()

            noisy_input_ids = input_ids.clone()

            input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
            b, l = input_ids.shape
            t = torch.rand((b,), device=input_ids.device)
            eps=1e-3
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask[:, None].repeat(1, l)

            mask_indices = torch.rand((b, l), device=input_ids.device) < p_mask
            x_t = torch.where(mask_indices, mask_id, input_ids).reshape(labels.shape)
            noisy_input_ids[labels != -100] = x_t[labels != -100]
            mask = (noisy_input_ids != mask_id)
            labels[mask] = -100
            input_ids = torch.cat([noisy_input_ids, input_ids.reshape(labels.shape)], dim=1)

            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_labels = original_labels.clone()

            complementary_input_ids = original_input_ids.reshape(original_input_ids.shape[0] * original_input_ids.shape[1] // self.model.bd_size, self.model.bd_size)

            complementary_mask_indices = ~mask_indices
            complementary_x_t = torch.where(complementary_mask_indices, mask_id, complementary_input_ids).reshape(labels.shape)
            complementary_noisy_input_ids[complementary_labels != -100] = complementary_x_t[complementary_labels != -100]
            complementary_mask = (complementary_noisy_input_ids != mask_id)
            complementary_labels[complementary_mask] = -100
            complementary_input_ids = torch.cat([complementary_noisy_input_ids, complementary_input_ids.reshape(complementary_labels.shape)], dim=1)

            input_ids = torch.cat([input_ids, complementary_input_ids], dim=0)
            labels = torch.cat([labels, complementary_labels], dim=0)
            
        # --- Fast-dLLM Diffusion Logic End ---

        # Model Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, # DeltaNet 通常忽略这个
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True, # 需要 hidden states 来算 logits
            return_dict=True,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state

        # Compute Logits
        if self.training:
            hidden_states = hidden_states[:, :hidden_states.shape[1]//2, :]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Standard Cross Entropy
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndBlockCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            block_past_key_values=None,
        )

    # -------------------------------------------------------------------------
    # Generate 函数：必须保留 diffusion 迭代生成的逻辑
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens, 
        mask_id=151665,
        threshold=1,
        small_block_size=8,
        block_size=64,
        stop_token=151645,
        stopping_criteria=None,
        top_p=0.95,
        temperature=0,
        use_block_cache=False,
        **kwargs
    ):
        num_blocks = max_new_tokens // block_size
        original_input_length = input_ids.shape[1]

        if input_ids.shape[1] > block_size:
            output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, block_size=block_size,past_key_values=None)
            logits, past_key_values = output.logits, output.past_key_values
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]
            # Initialize x_init with mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate the next token
                if mask_idx.sum() == 0:
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break


                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=False, past_key_values=past_key_values,block_size=block_size).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]


                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        # Select tokens with probability greater than threshold from p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

            input_ids = x_t
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        return input_ids

    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        # Calculate probabilities
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            p_1t = torch.softmax(logits, dim=-1)
            x_1 = p_1t.argmax(dim=-1)
            return x_1, p_1t
                            
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        probs[indices_to_remove] = 0

        # Renormalize so that the probabilities of remaining tokens sum to 1
        # Add a small epsilon value to prevent division by zero
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        normalized_probs = probs / probs_sum

        p_1t = normalized_probs
        x_1 = torch.multinomial(p_1t[0], num_samples=1).unsqueeze(0).squeeze(-1)

        return x_1, p_1t