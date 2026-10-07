"""Autocast changes forward operations, never parameter or optimizer dtypes."""
from contextlib import nullcontext

import torch


def precision_dtype(settings):
    name = settings.get("mixed_precision", "fp32")
    if name not in {"fp32", "fp16", "bf16"}:
        raise ValueError("model.mixed_precision must be fp32, fp16 or bf16")
    return {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def autocast_context(settings, device):
    dtype = precision_dtype(settings)
    return nullcontext() if dtype is None else torch.autocast(
        device.type, dtype=dtype, cache_enabled=settings.get("autocast_cache_enabled", False))


def build_scaler(settings, device):
    # A smaller initial scale avoids repeated warmup skips for the RLCD head.
    return torch.amp.GradScaler(device.type, init_scale=settings.get("fp16_initial_scale", 1024.),
                                enabled=precision_dtype(settings) == torch.float16)


def optimizer_step(optimizer, parameters, clip_grad, scaler):
    """Unscale once after all microbatches, then clip and update once."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad,
                                         error_if_nonfinite=not scaler.is_enabled(), foreach=False)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= scale_before
    return norm.item() if torch.isfinite(norm) else None, updated
