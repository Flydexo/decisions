from __future__ import annotations

import os
import random
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
    payload = {"format_version": 1, "head": model.head_state(),
               "optimizer": cpu_tree(optimizer.state_dict()) if include_optimizer else None, "config": config,
               "progress": dict(progress), "rng": rng_state(device), "device_type": device.type}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    # Checkpoints produced by this package contain plain dicts/tensors/primitive RNG states.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return data


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
    for key in ("train_pool_rows", "train_cache_dir"):
        if old.get("data", {}).get(key) != config.get("data", {}).get(key):
            raise ValueError(f"Resume would change data.{key}")
    if data["device_type"] != device.type:
        raise ValueError("Exact RNG resume requires the same device type")
    model.load_head(data["head"])
    optimizer.load_state_dict(data["optimizer"])
    restore_rng(data["rng"], device)
    progress = dict(data["progress"])
    if any(old["evaluation"].get(key) != config["evaluation"].get(key)
           for key in ("max_rows", "shuffle_buffer", "strict")):
        progress["best_accuracy"] = -1.0
        print("Evaluation sampling changed; resetting best-checkpoint accuracy.", flush=True)
    return progress
