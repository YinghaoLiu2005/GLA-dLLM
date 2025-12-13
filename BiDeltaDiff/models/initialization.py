import torch
import re

def _candidate_old_keys_for_new_key(new_key: str) -> list[str]:
    """
    Map new attention keys to possible old checkpoint keys.
    You can extend this list based on what you find in old_state_dict.keys().
    """
    candidates = [new_key]

    # Map:
    # model.layers.N.attn.fwd_attn.q_proj.weight  -> model.layers.N.self_attn.q_proj.weight
    # model.layers.N.attn.bwd_attn.q_proj.weight  -> model.layers.N.self_attn.q_proj.weight
    m = re.match(r"^(model\.layers\.\d+)\.attn\.(fwd_attn|bwd_attn)\.(q_proj|k_proj|v_proj)\.weight$", new_key)
    if m:
        prefix, _, proj = m.groups()
        candidates.extend([
            f"{prefix}.self_attn.{proj}.weight",
            f"{prefix}.attn.{proj}.weight",
            f"{prefix}.attention.{proj}.weight",
        ])

    # If you have a "model.layers.N.attn.q_proj.weight" in the new model too
    m2 = re.match(r"^(model\.layers\.\d+)\.attn\.(q_proj|k_proj|v_proj)\.weight$", new_key)
    if m2:
        prefix, proj = m2.groups()
        candidates.extend([
            f"{prefix}.self_attn.{proj}.weight",
            f"{prefix}.attn.{proj}.weight",
            f"{prefix}.attention.{proj}.weight",
        ])

    # Deduplicate but keep order
    seen = set()
    out = []
    for k in candidates:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def load_partial_weights(
    new_model,
    old_state_dict,
    verbose=True,
    only_print_first_layer: bool = True,
    layer_idx: int = 0,
):
    new_state_dict = new_model.state_dict()

    loaded = []              # list of new keys loaded
    loaded_from = {}         # new_key -> old_key used
    skipped_missing = []
    skipped_shape = []       # (new_key, new_shape, old_shape, tried_old_key)

    for new_key, new_param in new_state_dict.items():
        loaded_flag = False
        tried = _candidate_old_keys_for_new_key(new_key)

        for old_key in tried:
            if old_key not in old_state_dict:
                continue
            old_param = old_state_dict[old_key]
            if old_param.shape != new_param.shape:
                skipped_shape.append((new_key, tuple(new_param.shape), tuple(old_param.shape), old_key))
                continue

            new_state_dict[new_key] = old_param
            loaded.append(new_key)
            loaded_from[new_key] = old_key
            loaded_flag = True
            break

        if not loaded_flag:
            # none of candidates exist w/ matching shape
            if any(k in old_state_dict for k in tried):
                # existed but shape mismatch (already recorded), still mark as missing for summary? no
                pass
            else:
                skipped_missing.append(new_key)

    new_model.load_state_dict(new_state_dict)

    if verbose:
        # Filter to layer_idx for printing
        patterns = [
            rf"(^|\.)(layers|layer)\.{layer_idx}\.",
            rf"(^|\.)(model)\.(layers|layer)\.{layer_idx}\.",
            rf"(^|\.)(transformer)\.(h|layers)\.{layer_idx}\.",
        ]
        layer_re = re.compile("|".join(f"(?:{p})" for p in patterns))

        loaded_p = [k for k in loaded if layer_re.search(k)]
        missing_p = [k for k in skipped_missing if layer_re.search(k)]
        shape_p = [x for x in skipped_shape if layer_re.search(x[0])]

        print(f"\n========== Loaded Weights (layer {layer_idx} only) ==========\n")
        for k in loaded_p:
            src = loaded_from.get(k, k)
            if src != k:
                print(f"{k}  <=  {src}")
            else:
                print(k)

        print(f"\n========== Skipped (missing key) (layer {layer_idx} only) ==========\n")
        for k in missing_p:
            print(k)

        print(f"\n========== Skipped (shape mismatch) (layer {layer_idx} only) ==========\n")
        # show only first few to avoid huge spam
        for new_key, new_shape, old_shape, old_key in shape_p[:200]:
            print(f"{new_key}  !=shape  {old_key} | new={new_shape} old={old_shape}")

        print(
            f"\n[load_partial_weights] layer {layer_idx}: "
            f"loaded={len(loaded_p)}, missing={len(missing_p)}, shape_mismatch={len(shape_p)} "
            f"(total loaded={len(loaded)}, total missing={len(skipped_missing)}, total shape_mismatch={len(skipped_shape)})\n"
        )

    return new_model
