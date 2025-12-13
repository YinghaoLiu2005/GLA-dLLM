import torch
import re

def load_partial_weights(new_model, old_state_dict, verbose=True, only_print_first_layer: bool = True, layer_idx: int = 0):
    new_state_dict = new_model.state_dict()
    loaded, skipped = [], []
    import re
    for name, param in new_state_dict.items():
        if name in old_state_dict and old_state_dict[name].shape == param.shape:
            new_state_dict[name] = old_state_dict[name]
            loaded.append(name)
        else:
            skipped.append(name)

    new_model.load_state_dict(new_state_dict)

    if verbose:
        if only_print_first_layer:
            # Match common layer naming conventions:
            # - layers.0.*
            # - model.layers.0.*
            # - transformer.h.0.*
            patterns = [
                rf"(^|\.)(layers|layer)\.{layer_idx}\.",          # layers.0.
                rf"(^|\.)(model)\.(layers|layer)\.{layer_idx}\.", # model.layers.0.
                rf"(^|\.)(transformer)\.(h|layers)\.{layer_idx}\."# transformer.h.0.
            ]
            layer_re = re.compile("|".join(f"(?:{p})" for p in patterns))

            loaded_to_print = [k for k in loaded if layer_re.search(k)]
            skipped_to_print = [k for k in skipped if layer_re.search(k)]

            print(f"\n========== Loaded Weights (layer {layer_idx} only) ==========\n")
            for k in loaded_to_print:
                print(k)

            print(f"\n========== Skipped Weights (layer {layer_idx} only) ==========\n")
            for k in skipped_to_print:
                print(k)

            # Optional: show counts to know if filter matched anything
            print(
                f"\n[load_partial_weights] layer {layer_idx}: "
                f"loaded={len(loaded_to_print)}, skipped={len(skipped_to_print)} "
                f"(total loaded={len(loaded)}, total skipped={len(skipped)})\n"
            )
        else:
            print("\n================== Loaded Weights =====================\n")
            for k in loaded:
                print(k)
            print("\n================ Skipped Weights ====================\n")
            for k in skipped:
                print(k)

    return new_model
