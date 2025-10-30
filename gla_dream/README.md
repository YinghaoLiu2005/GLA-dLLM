# GLADream: GLA-based Block-Diffusion LLM

This module implements the GLADream architecture based on the Fast-dLLM v2 paper and Dream's diffusion approach.

## Architecture

The GLADream model consists of multiple layers, each containing:

1. **Inter-block Recurrent Unit**: A unidirectional GLA layer for propagating context between blocks
2. **Intra-block Bidirectional Unit**: A bidirectional GLA block that processes each block with forward and backward scans
3. **Feed-Forward Network (FFN)**: Standard MLP

### Information Flow

- Each block receives a global recursive state S_{i-1} from the previous block
- The state is injected into the current block (global or boundary mode)
- The block is processed bidirectionally using BiDirectionalGLABlock
- Output flows through FFN
- The block's output is used to compute a new global state S_i for the next block

### Gating Mechanisms

**Intra-block fusion**: Uses a gating vector to combine forward and backward hidden states
```
gate_input = concat(h_t_forward, h_t_backward)
g_t = sigmoid(linear_gate(gate_input))
h_t_final = g_t * h_t_forward + (1 - g_t) * h_t_backward
```

**Inter-block fusion**: Uses a gating vector for state propagation between blocks
```
gate_input = concat(h_last_forward, h_first_backward)
gate = sigmoid(linear_gate_inter_block(gate_input))
next_state = gate * h_last_forward + (1 - gate) * h_first_backward
```

By default, the gate weights for inter-block and intra-block fusion are **not shared** (configurable via `share_inter_intra_gate_weights`).

## Configuration

Key configuration parameters:

- `block_size`: Block size for block-wise processing (default: 32)
- `expand_k`, `expand_v`: Expansion factors for GLA (default: 1)
- `attn_mode`: GLA attention mode
- `inter_block_injection_mode`: How to inject inter-block state ('global' or 'boundary', default: 'global')
- `share_inter_intra_gate_weights`: Whether to share gate weights (default: False)
- `use_complementary_mask`: Use Fast-dLLM v2 complementary mask vs Dream mask (default: False)

## Training

Supports two training strategies:

1. **Dream mask strategy** (default): Standard diffusion training approach
2. **Fast-dLLM v2 complementary mask** (`use_complementary_mask=True`): Uses complementary masking for parallel block processing

## Inference

Block-wise recursive generation:

1. **Prompt processing**: Process the prompt through parallel mode to get initial state S_prompt
2. **Block-by-block generation**: For each new block i:
   - Get historical state: S_{i-1}
   - Initialize with [MASK] tokens
   - Iterative denoising:
     * Inject S_{i-1} into current_block
     * Process through bidirectional GLA (parallel for entire block)
     * Decode high-confidence tokens
     * Update current_block
     * Repeat until complete
   - Update global state via inter-block recurrent unit

## Usage

```python
from gla_dream import GLADreamConfig, GLADreamModel

# Create config
config = GLADreamConfig(
    vocab_size=151936,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    block_size=32,
    expand_k=1,
    expand_v=1,
)

# Create model
model = GLADreamModel(config)

# Forward pass
outputs = model(input_ids, labels=labels)

# Generation
generated = model.generate(input_ids, max_length=100, threshold=0.9)
```

## File Structure

- `__init__.py`: Module exports
- `configuration_gla_dream.py`: Model configuration
- `modeling_gla_dream.py`: Core model implementation
- `generation_utils_gla_dream.py`: Generation utilities
- `tokenization_gla_dream.py`: Tokenizer (reuses Dream tokenizer)

## Dependencies

- PyTorch
- Transformers
- FLA (for GatedLinearAttention)

## References

- Fast-dLLM v2: Efficient Block-Diffusion LLM (arXiv:2509.26328)
- Dream: Diffusion-based Progressive Denoising for LLM Generation

