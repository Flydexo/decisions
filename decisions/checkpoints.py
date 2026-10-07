from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def rng_state(device):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state().cpu()
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, config, progress, device, include_optimizer=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finetuning = model.encoder_trainable
    # torch.save transfers accelerator storages individually. Avoid a second
    # complete CPU copy of multi-GiB encoder weights and Adam states.
    payload = {"format_version": 2 if finetuning else 1, "head": model.head_state(cpu=not finetuning),
               "optimizer": (optimizer.state_dict() if finetuning else cpu_tree(optimizer.state_dict())) if include_optimizer else None, "config": config,
               "progress": dict(progress), "rng": rng_state(device), "device_type": device.type}
    if finetuning:
        payload["encoder"] = {k: v.detach() for k, v in model.bert.state_dict().items()}
    scaler = getattr(optimizer, "_decision_scaler", None)
    if include_optimizer and scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    reserve = config.get("training", {}).get("min_free_disk_gib", 0) * 2**30
    required = checkpoint_tensor_bytes(payload) + 10 * 2**20 + reserve
    if shutil.disk_usage(path.parent).free < required:
        raise RuntimeError(f"Checkpoint needs approximately {required / 2**30:.2f} GiB free, including disk reserve")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(checkpoint_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(checkpoint_tensor_bytes(v) for v in value)
    return 0


def read_checkpoint(path):
    # Checkpoints produced by this package contain plain dicts/tensors/primitive RNG states.
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if data.get("format_version") not in (1, 2):
        raise ValueError("Unsupported checkpoint version")
    return data


def load_checkpoint_model(model, data):
    if data.get("encoder") is not None:
        model.bert.load_state_dict(data["encoder"], strict=True)
    elif model.encoder_trainable:
        raise ValueError("An unfrozen encoder checkpoint must contain its trained encoder weights")
    model.load_head(data["head"])


def resume_checkpoint(path, model, optimizer, config, datasets, device):
    data = read_checkpoint(path)
    old = data["config"]
    if data["optimizer"] is None:
        raise ValueError("This is an inference-only best checkpoint; resume training from last.pt")
    for key in ("model", "ablation", "dataset_configs", "seed"):
        current = datasets if key == "dataset_configs" else config[key]
        if old[key] != current:
            raise ValueError(f"Resume would change {key}; start a new run instead")
    for key in ("batch_size", "shuffle_buffer", "learning_rate", "weight_decay", "sigma", "candidates", "clip_grad"):
        if old["training"][key] != config["training"][key]:
            raise ValueError(f"Resume would change training.{key}")
    for key in ("warmup_steps", "decay_steps", "min_learning_rate_ratio", "validation_role"):
        if old["training"].get(key) != config["training"].get(key):
            raise ValueError(f"Resume would change training.{key}")
    for key, default in (("gradient_accumulation_steps", 1), ("question_microbatch_size", 0),
                         ("encoder_learning_rate", None), ("optimizer_foreach", None)):
        if old["training"].get(key, default) != config["training"].get(key, default):
            raise ValueError(f"Resume would change training.{key}")
    for key in ("train_pool_rows", "train_cache_dir"):
        if old.get("data", {}).get(key) != config.get("data", {}).get(key):
            raise ValueError(f"Resume would change data.{key}")
    if data["device_type"] != device.type:
        raise ValueError("Exact RNG resume requires the same device type")
    load_checkpoint_model(model, data)
    optimizer.load_state_dict(data["optimizer"])
    scaler = getattr(optimizer, "_decision_scaler", None)
    if scaler is not None and scaler.is_enabled():
        if "scaler" not in data:
            raise ValueError("FP16 resume requires the saved gradient scaler state")
        scaler.load_state_dict(data["scaler"])
    if device.type != "cpu":
        # Adam's CPU step scalars otherwise retain the entire memory-mapped
        # checkpoint after its moment tensors have been copied to the GPU.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.ndim == 0:
                    state[key] = value.clone()
    restore_rng(data["rng"], device)
    progress = dict(data["progress"])
    if any(old["evaluation"].get(key) != config["evaluation"].get(key)
           for key in ("max_rows", "shuffle_buffer", "strict")):
        progress["best_accuracy"] = -1.0
        print("Evaluation sampling changed; resetting best-checkpoint accuracy.", flush=True)
    return progress
