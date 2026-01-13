"""BiDeltaDiff model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BiDeltaDiffConfig(PretrainedConfig):

    model_type = "BiDeltaDiff"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `BiDeltaDiff`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_heads=12,
        num_key_value_heads=12,
        head_dim=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        # --- KDA 特有 Tricks (保留) ---
        attn_mode: str = "chunk",       # 必须是 chunk 模式以加速
        use_short_conv: bool = False,    # [保留] 局部卷积，增强局部性
        conv_size: int = 4,             # 卷积核大小
        expand_v: float = 1.0,          # [保留] Value 扩展系数，先设 1.0
        use_l2warp: bool = False,       # KDA 的数值稳定性 Trick，默认关，除非你明确知道怎么用
        allow_neg_eigval: bool = False,
        is_bidirectional=True,
        chunk_size=64,
        use_cache=False,
        bd_size=64,

        # --- 优化与初始化 ---
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,

        
        # --- 混合架构接口 (暂时留空) ---
        attn: dict | None = None,       # 默认纯 DeltaNet

        rope_theta=None,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=None,
        layer_types=None,
        attention_dropout=0.0,

        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads  
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.chunk_size = chunk_size
        self.rms_norm_eps = rms_norm_eps
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.expand_v = expand_v
        self.use_l2warp = use_l2warp
        self.allow_neg_eigval = allow_neg_eigval
        
        self.is_bidirectional = is_bidirectional
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.attn = attn
        self.bd_size = bd_size


        if self.num_heads * self.head_dim != self.hidden_size:
            pass 

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )