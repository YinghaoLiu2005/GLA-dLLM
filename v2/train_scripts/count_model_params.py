#!/usr/bin/env python
"""Utility script to report parameter counts and memory footprint for a HF-compatible model checkpoint.

Example:
    python count_model_params.py --model_name_or_path /path/to/checkpoint --trust_remote_code
"""

import argparse
import math
import os
import sys
from typing import Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# --------------------------------------------------------------------------------------
# Optional: register custom GLADream config/model so Auto* can discover the checkpoint.
# --------------------------------------------------------------------------------------
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    candidate_paths = [
        os.path.join(current_dir, '..'),              # v2/
        repo_root,                                   # repository root
        os.path.join(current_dir, '..', 'gla_dream'),  # legacy relative location (kept for safety)
        os.path.join(repo_root, 'gla_dream'),          # explicit package path
    ]

    for path in candidate_paths:
        path = os.path.abspath(path)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)

    from gla_dream.configuration_gla_dream import GLADreamConfig
    from gla_dream.modeling_gla_dream import GLADreamModel

    try:
        AutoConfig.register("GLADream", GLADreamConfig)
    except ValueError:
        pass  # Already registered in this interpreter session.

    try:
        AutoModelForCausalLM.register(GLADreamConfig, GLADreamModel)
    except ValueError:
        pass
except Exception as register_exc:
    # If GLADream is not available this registration step silently fails; standard
    # models are unaffected, but warn for easier debugging.
    print(f"[WARN] Could not register GLADream with Auto classes: {register_exc}")
    GLADreamConfig = None  # type: ignore
    GLADreamModel = None  # type: ignore


def human_readable(num: float, suffix: str) -> str:
    """Format numbers (e.g. parameters, bytes) with binary prefixes."""
    if num == 0:
        return f"0{suffix}"
    units = ["", "K", "M", "B", "T", "P"]
    k = 1000.0
    magnitude = int(math.floor(math.log(num, k)))
    magnitude = min(magnitude, len(units) - 1)
    scaled = num / (k ** magnitude)
    return f"{scaled:.3f}{units[magnitude]}{suffix}"


def collect_param_stats(model) -> dict[str, float]:
    total_params = 0
    trainable_params = 0
    dtype_buckets: dict[torch.dtype, int] = {}

    for param in model.parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count

        dtype_buckets[param.dtype] = dtype_buckets.get(param.dtype, 0) + count * param.element_size()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "dtype_bytes": dtype_buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report parameter counts for a model checkpoint.")
    parser.add_argument("--model_name_or_path", required=True, help="Path or identifier of the model checkpoint.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow custom model code to be executed.")
    parser.add_argument("--revision", default=None, help="Optional model revision (branch/tag).")

    args = parser.parse_args()

    # Attempt to load the config first to surface registration issues early.
    print(f"[INFO] Loading config from {args.model_name_or_path} ...")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"[INFO] Detected model type: {getattr(config, 'model_type', 'unknown')}\n")

    # Try to instantiate the model on the meta device to avoid materialising weights in memory.
    load_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        low_cpu_mem_usage=True,
    )

    model = None
    loaded_on_meta = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="meta",
            **load_kwargs,
        )
        loaded_on_meta = True
        print("[INFO] Loaded model on the meta device (no weights materialised).")
    except Exception as meta_exc:
        print(f"[WARN] Could not load on meta device: {meta_exc}\n[INFO] Falling back to CPU load (this may use significant RAM).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
            **load_kwargs,
        )

    stats = collect_param_stats(model)

    total = stats["total_params"]
    trainable = stats["trainable_params"]
    frozen = stats["frozen_params"]

    print("\n=== Parameter Counts ===")
    print(f"Total parameters        : {total:,} ({human_readable(total, '')})")
    print(f"Trainable parameters    : {trainable:,} ({human_readable(trainable, '')})")
    print(f"Frozen parameters       : {frozen:,} ({human_readable(frozen, '')})")

    print("\n=== Memory Footprint by dtype (assuming dense storage) ===")
    total_bytes = 0
    for dtype, bytes_used in stats["dtype_bytes"].items():
        total_bytes += bytes_used
        print(f"{str(dtype):>12}: {bytes_used:,} bytes ({human_readable(bytes_used, 'B')})")

    print(f"\nApproximate parameter storage: {total_bytes:,} bytes ({human_readable(total_bytes, 'B')})")

    if loaded_on_meta:
        print("\n[INFO] Model was loaded on the meta device. No weight tensors were materialised; results represent the parameter shapes recorded in the checkpoint.")



if __name__ == "__main__":
    main()
