import inspect

import torch
from fairseq import checkpoint_utils


def load_trusted_model_ensemble_and_task(paths, suffix=""):
    original_torch_load = torch.load
    torch_load_params = inspect.signature(original_torch_load).parameters

    def torch_load_compat(*args, **kwargs):
        if "weights_only" in torch_load_params and "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = torch_load_compat
        return checkpoint_utils.load_model_ensemble_and_task(paths, suffix=suffix)
    finally:
        torch.load = original_torch_load
