import re
import torch
from transformers import AutoModel

fast_dllm_model_path = "/data/yinghaoliu/GLA-dLLM/trained_models/Fast_dLLM_v2_1.5B"
fast_dllm_7B_model_path = "/data/yinghaoliu/GLA-dLLM/trained_models/Fast_dLLM_v2_7B"
def extract_layer_indices_from_keys(keys):
    """Return sorted list of integer layer indices found in keys like 'layers.{i}.'"""
    idxs = set()
    pattern = re.compile(r"layers\.(\d+)\.")
    for k in keys:
        m = pattern.search(k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)

def write_all_keys(state_dict, out_path):
    """Write all keys and their shapes from a state_dict to out_path."""
    with open(out_path, "w") as f:
        for k, v in state_dict.items():
            try:
                shape = tuple(v.shape)
            except Exception:
                # v might be a meta value or not a tensor
                shape = None
            f.write(f"{k} \t {shape}\n")


def group_state_dict(state_dict):
    """Group state_dict keys into modules: embeddings, lm_head, norm, layers (per-index), others."""
    groups = {
        "embeddings": [],
        "lm_head": [],
        "norm": [],
        "layers": {},  # idx -> {subgroup -> list}
        "other": [],
    }

    for k, v in state_dict.items():
        parts = k.split('.')
        # detect layer keys like model.layers.0.mlp.gate_proj.weight or model.layers.10.self_attn.q_proj.weight
        if 'layers' in parts:
            try:
                li = parts.index('layers')
                idx = int(parts[li + 1])
                rest = parts[li + 2:]
            except Exception:
                groups['other'].append((k, v))
                continue

            layer = groups['layers'].setdefault(idx, {})
            # categorize by rest[0]
            if len(rest) == 0:
                layer.setdefault('unknown', []).append((k, v))
                continue
            head = rest[0]
            if head in ('mlp', 'mlp_proj', 'mlp_block'):
                layer.setdefault('mlp', []).append((k, v))
            elif 'attn' in head or 'self_attn' in head or 'gla' in head or head in ('q_proj','k_proj','v_proj','o_proj'):
                layer.setdefault('attention', []).append((k, v))
            elif 'norm' in head or 'layernorm' in head or 'input_layernorm' in head or 'post_attention_layernorm' in head or 'post_intra_layernorm' in head:
                layer.setdefault('layernorm', []).append((k, v))
            else:
                # other within layer
                layer.setdefault('other', []).append((k, v))
        else:
            # Non-layer top-level
            if 'embed' in k or 'embedding' in k or 'embed_tokens' in k:
                groups['embeddings'].append((k, v))
            elif k.startswith('lm_head') or '.lm_head' in k:
                groups['lm_head'].append((k, v))
            elif k.endswith('.weight') and ('.norm' in k or 'norm' in k or 'RMSNorm' in k):
                groups['norm'].append((k, v))
            else:
                groups['other'].append((k, v))

    return groups


def write_grouped(state_dict, out_path):
    groups = group_state_dict(state_dict)
    with open(out_path, 'w') as f:
        # Embeddings
        f.write('=== EMBEDDINGS ===\n')
        for k, v in groups['embeddings']:
            try:
                shape = tuple(v.shape)
            except Exception:
                shape = None
            f.write(f"{k}\t{shape}\n")
        f.write('\n')

        # LM head
        f.write('=== LM_HEAD ===\n')
        for k, v in groups['lm_head']:
            try:
                shape = tuple(v.shape)
            except Exception:
                shape = None
            f.write(f"{k}\t{shape}\n")
        f.write('\n')

        # Norm / global norms
        f.write('=== GLOBAL_NORMS ===\n')
        for k, v in groups['norm']:
            try:
                shape = tuple(v.shape)
            except Exception:
                shape = None
            f.write(f"{k}\t{shape}\n")
        f.write('\n')

        # Layers
        f.write('=== LAYERS ===\n')
        for idx in sorted(groups['layers'].keys()):
            f.write(f'-- Layer {idx} --\n')
            layer = groups['layers'][idx]
            for subgroup in ('attention', 'mlp', 'layernorm', 'other', 'unknown'):
                if subgroup in layer:
                    f.write(f'  [{subgroup}]\n')
                    for k, v in layer[subgroup]:
                        try:
                            shape = tuple(v.shape)
                        except Exception:
                            shape = None
                        f.write(f"    {k}\t{shape}\n")
            f.write('\n')

        # Other
        f.write('=== OTHER ===\n')
        for k, v in groups['other']:
            try:
                shape = tuple(v.shape)
            except Exception:
                shape = None
            f.write(f"{k}\t{shape}\n")



def main():
    
    # 1) Fast-dLLM checkpoint (if available)
    try:
        base_model = AutoModel.from_pretrained(fast_dllm_model_path)
        base_sd = base_model.state_dict()
        write_all_keys(base_sd, "fast_dllm_all_keys.txt")
        write_grouped(base_sd, "fast_dllm_grouped.txt")
        print("Wrote fast_dllm_all_keys.txt and fast_dllm_grouped.txt")
    except Exception as e:
        print("Failed to load Fast_dLLM from", fast_dllm_model_path, "error:", e)

    # 2) GLADream model (construct and inspect state_dict)
    try:
        from gla_dream.configuration_gla_dream import GLADreamConfig
        from gla_dream.modeling_gla_dream import GLADreamModel

        cfg = GLADreamConfig()
        gla = GLADreamModel(cfg)
        write_all_keys(gla.state_dict(), "gla_all_keys.txt")
        write_grouped(gla.state_dict(), "gla_grouped.txt")
        print("Wrote gla_all_keys.txt and gla_grouped.txt")
    except Exception as e:
        print("Failed to instantiate GLADreamModel:", e)

    # 3) Dream model (construct and inspect state_dict)
    try:
        from dream.model.modeling_dream import DreamConfig, DreamModel

        dcfg = DreamConfig()
        dream = DreamModel(dcfg)
        write_all_keys(dream.state_dict(), "dream_all_keys.txt")
        write_grouped(dream.state_dict(), "dream_grouped.txt")
        print("Wrote dream_all_keys.txt and dream_grouped.txt")
    except Exception as e:
        print("Failed to instantiate DreamModel:", e)

    # 1) Fast-dLLM-7B checkpoint (if available)
    try:
        base_model = AutoModel.from_pretrained(fast_dllm_7B_model_path)
        base_sd = base_model.state_dict()
        write_all_keys(base_sd, "fast_dllm7B_all_keys.txt")
        write_grouped(base_sd, "fast_dllm7B_grouped.txt")
        print("Wrote fast_dllm7B_all_keys.txt and fast_dllm7B_grouped.txt")
    except Exception as e:
        print("Failed to load Fast_dLLM-7B from", fast_dllm7B_model_path, "error:", e)




if __name__ == '__main__':
    main()

