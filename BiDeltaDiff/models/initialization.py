import torch
import re
from datasets import Dataset as HFDataset, DatasetDict
import torch.nn as nn

def _candidate_old_keys_for_new_key(new_key: str) -> list[str]:
    """
    Map new attention keys to possible old checkpoint keys.
    You can extend this list based on what you find in old_state_dict.keys().
    """
    candidates = [new_key]

    # --- 1. Global Keys Mapping (Embeddings, Head, Norm) ---
    if "embed_tokens" in new_key:
        candidates.append("model.embed_tokens.weight")
    elif "lm_head" in new_key:
        candidates.append("lm_head.weight")
    elif new_key == "model.norm.weight" or new_key == "norm.weight":
        candidates.append("model.norm.weight")

    # --- 2. Layer-wise Mapping ---
    # Map: model.layers.N.attn.fwd_attn.q_proj  -> model.layers.N.self_attn.q_proj
    # Map: model.layers.N.attn_norm             -> model.layers.N.input_layernorm
    
    # Capture prefix like "model.layers.0"
    layer_match = re.match(r"^(model\.layers\.\d+)\.(.*)$", new_key)
    if layer_match:
        prefix, suffix = layer_match.groups()
        
        # A. Attention Projection Mapping
        # Matches: attn.fwd_attn.q_proj.weight OR attn.bwd_attn.q_proj.weight
        m_attn = re.match(r"^attn\.attn\.(q_proj|k_proj|v_proj)\.weight$", suffix)
        if m_attn:
            proj = m_attn.group(1)
            candidates.append(f"{prefix}.self_attn.{proj}.weight")
        
        # B. LayerNorm Mapping
        if suffix == "attn_norm.weight":
            candidates.append(f"{prefix}.input_layernorm.weight")
        if suffix == "mlp_norm.weight":
            candidates.append(f"{prefix}.post_attention_layernorm.weight")

    # Deduplicate but keep order
    seen = set()
    out = []
    for k in candidates:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def _init_fusion_layer(param, key_name):
    # 简单的初始化辅助函数
    if param.dim() < 2:
        return None
    # 如果是新增加的参数（如 a_proj, b_proj, g_proj, conv1d 等），可以在这里做特殊初始化
    # 目前保持默认（随机）
    return None

def load_partial_weights(
    new_model,
    old_state_dict,
    head_dim=128,  # [新增参数] 需要知道 head_dim 才能正确切分 GQA
    verbose=True,
    only_print_first_layer: bool = True,
    layer_idx: int = 0,
    init_missing: bool = True,
):
    new_state_dict = new_model.state_dict()

    loaded = []              # list of new keys loaded
    loaded_from = {}         # new_key -> old_key used
    skipped_missing = []
    skipped_shape = []       # (new_key, new_shape, old_shape, tried_old_key)
    expanded_keys = []       # [新增] 记录被扩展的 key
    custom_initialized = []  # the key initialize by ourself
    for new_key, new_param in new_state_dict.items():
        loaded_flag = False
        tried = _candidate_old_keys_for_new_key(new_key)

        for old_key in tried:
            if old_key not in old_state_dict:
                continue
            old_param = old_state_dict[old_key]
            
            # --- Case 1: Perfect Shape Match ---
            if old_param.shape == new_param.shape:
                new_state_dict[new_key] = old_param
                loaded.append(new_key)
                loaded_from[new_key] = old_key
                loaded_flag = True
                break
            
            # --- Case 2: GQA Expansion (k_proj, v_proj) ---
            # 如果是 K 或 V，且新维度是旧维度的整数倍，则执行复制扩展
            elif ("k_proj" in new_key or "v_proj" in new_key) and \
                len(old_param.shape) >= 2 and len(new_param.shape) >= 2 and \
                 (new_param.shape[1] == old_param.shape[1]) and \
                 (new_param.shape[0] > old_param.shape[0]) and \
                 (new_param.shape[0] % old_param.shape[0] == 0):
                
                ratio = new_param.shape[0] // old_param.shape[0]
                
                # Reshape logic: [n_kv * head_dim, hidden] -> [n_kv, head_dim, hidden]
                n_kv_old = old_param.shape[0] // head_dim
                hidden_size = old_param.shape[1]
                
                # 1. View as heads
                w_reshaped = old_param.view(n_kv_old, head_dim, hidden_size)
                # 2. Repeat heads: [n_kv, ratio, head_dim, hidden]
                w_expanded = w_reshaped.unsqueeze(1).repeat(1, ratio, 1, 1)
                # 3. Flatten back: [n_kv * ratio * head_dim, hidden] -> [1536, 1536]
                w_final = w_expanded.reshape(-1, hidden_size)
                
                new_state_dict[new_key] = w_final
                loaded.append(new_key)
                loaded_from[new_key] = f"{old_key} (Expanded x{ratio})"
                expanded_keys.append(new_key)
                loaded_flag = True
                break
            
            else:
                # 记录最后一次尝试的 mismatched shape
                if old_key == tried[-1]: 
                    skipped_shape.append((new_key, tuple(new_param.shape), tuple(old_param.shape), old_key))
                continue

        if not loaded_flag:
            # 只有当所有候选 key 都不匹配或不存在时，才算 missing
            # 如果是因为 shape mismatch 被上面捕获了，这里其实也会走到，
            # 但我们在 summary 打印时主要看 skipped_missing 和 skipped_shape
            # 简单的逻辑：没 loaded 就是 missing (或者 shape mismatch 导致没 load)
            if init_missing:
                init_desc=_init_fusion_layer(new_state_dict[new_key], new_key)
                if init_desc:
                    custom_initialized.append(f"{new_key} -> {init_desc}")
            if not any(k in old_state_dict for k in tried):
                skipped_missing.append(new_key)

    new_model.load_state_dict(new_state_dict)

    if verbose:
        # [修改] 更新正则过滤器，允许全局参数通过
        patterns = [
            rf"(^|\.)(layers|layer)\.{layer_idx}\.",       # 匹配指定层
            r"embed_tokens",                                # 匹配 Embedding
            r"lm_head",                                     # 匹配 LM Head
            r"model\.norm",                                 # 匹配全局 Norm
        ]
        layer_re = re.compile("|".join(f"(?:{p})" for p in patterns))

        loaded_p = [k for k in loaded if layer_re.search(k)]
        missing_p = [k for k in skipped_missing if layer_re.search(k)]
        shape_p = [x for x in skipped_shape if layer_re.search(x[0])]

        print(f"\n========== Loaded Weights (Global + Layer {layer_idx}) ==========\n")
        for k in loaded_p:
            src = loaded_from.get(k, k)
            prefix = "[EXPANDED]" if k in expanded_keys else ""
            if src != k or prefix:
                print(f"{prefix} {k}  <=  {src}")
            else:
                print(k)

        print(f"\n========== Skipped (missing key) (Global + Layer {layer_idx}) ==========\n")
        for k in missing_p:
            print(k)

        print(f"\n========== Skipped (shape mismatch) (Global + Layer {layer_idx}) ==========\n")
        for new_key, new_shape, old_shape, old_key in shape_p[:50]:
            print(f"{new_key}  !=shape  {old_key} | new={new_shape} old={old_shape}")

        print(
            f"\n[load_partial_weights] Filtered View (Global + Layer {layer_idx}): "
            f"loaded={len(loaded_p)}, missing={len(missing_p)}, shape_mismatch={len(shape_p)} "
            f"(Total processed: loaded={len(loaded)}, missing={len(skipped_missing)})\n"
        )

    return new_model