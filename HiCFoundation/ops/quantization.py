import torch
import torch.nn as nn


def apply_int8_weight_only(model: nn.Module, skip_modules=None) -> nn.Module:
    """
    Weight-only INT8 quantization for nn.Linear layers using torchao.
    Only weights are quantized (to int8 + per-channel scale); activations
    stay in fp16/bf16. No calibration data required - safe for a model
    that was never trained with quantization in mind.

    skip_modules: iterable of module name substrings to leave in full
    precision (e.g. the small task-specific heads, if you want to protect
    them from precision loss).
    """
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    skip_modules = skip_modules or []

    def _filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        return not any(skip in fqn for skip in skip_modules)

    quantize_(model, Int8WeightOnlyConfig(), filter_fn=_filter_fn)
    return model


def quantize_model(model: nn.Module, mode: str = "none", skip_modules=None) -> nn.Module:
    """
    mode: "none" | "int8"
    Call this AFTER model.load_state_dict(...) and model.cuda(),
    BEFORE wrapping in nn.DataParallel.
    """
    if mode == "none":
        return model
    elif mode == "int8":
        return apply_int8_weight_only(model, skip_modules=skip_modules)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")