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

"""GLADream model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GLADreamConfig(PretrainedConfig):
    model_type = "GLADream"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        mask_token_id=151666,
        pad_token_id=151643,
        # GLA-specific parameters
        block_size=32,  # Block size for block-wise processing
        expand_k=1,  # Expansion factor for K in GLA
        expand_v=1,  # Expansion factor for V in GLA
        attn_mode='fwd_scan',  # GLA attention mode
        inter_block_injection_mode='global',  # How to inject inter-block state: 'global' or 'boundary'
        share_inter_intra_gate_weights=False,  # Whether to share gate weights between inter and intra block fusion
        use_complementary_mask=False,  # Use complementary mask strategy (Fast-dLLM v2) vs Dream mask
        qwen3_compatible: bool = True,  # compatible with qwen3 model
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        
        # GLA-specific configurations
        self.block_size = block_size
        # Backward-compat alias for training scripts expecting `bd_size`
        self.bd_size = block_size
        self.bdsize =block_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.attn_mode = attn_mode
        self.inter_block_injection_mode = inter_block_injection_mode
        self.share_inter_intra_gate_weights = share_inter_intra_gate_weights
        self.use_complementary_mask = use_complementary_mask
        self.qwen3_compatible = qwen3_compatible
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

