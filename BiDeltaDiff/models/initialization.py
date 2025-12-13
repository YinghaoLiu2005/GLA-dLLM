import torch

def load_partial_weights(new_model, old_state_dict, verbose=True):
    new_state_dict = new_model.state_dict()
    loaded, skipped = [], []

    for name, param in new_state_dict.items():
        if name in old_state_dict and old_state_dict[name].shape == param.shape:
            new_state_dict[name] = old_state_dict[name]
            loaded.append(name)
        else:
            skipped.append(name)

    new_model.load_state_dict(new_state_dict)

    if verbose:
        print("===== Loaded Weights =====")
        for k in loaded:
            print(k)
        print("\n===== Skipped Weights =====")
        for k in skipped:
            print(k)

    return new_model
