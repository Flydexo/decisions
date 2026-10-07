"""CUDA pilot checks and synthetic training inputs; no benchmark/test data."""
from __future__ import annotations

import platform
import shutil
from numbers import Real

import torch


def configure_cuda_budget(budget_gib, device):
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    if isinstance(budget_gib, bool) or not isinstance(budget_gib, Real) or not 0 < budget_gib * 2**30 <= total:
        raise ValueError("CUDA memory budget must be positive and fit the selected GPU")
    torch.cuda.set_per_process_memory_fraction(budget_gib * 2**30 / total, device)


def check_environment(config, run_dir):
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: use the RTX 4090 instance and a CUDA PyTorch build; no CPU/MPS fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if "4090" not in props.name:
        raise RuntimeError(f"This pilot targets RTX 4090; selected GPU is {props.name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device/runtime does not support BF16")
    budget = config["training"]["cuda_memory_budget_gib"]
    free, total = torch.cuda.mem_get_info(device)
    if total < 22 * 2**30 or free < budget * 2**30:
        raise RuntimeError(f"Need a full 24 GB GPU with at least {budget} GiB currently free; "
                           f"found {total / 2**30:.2f} total / {free / 2**30:.2f} free GiB")
    if shutil.disk_usage(run_dir).free < 20 * 2**30:
        raise RuntimeError("Need at least 20 GiB free for model download and atomic full-encoder checkpoints")
    configure_cuda_budget(budget, device)
    import transformers
    return {"gpu": props.name, "compute_capability": [props.major, props.minor],
            "total_gpu_gib": total / 2**30, "initial_free_gpu_gib": free / 2**30,
            "cuda_budget_gib": budget, "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "transformers": transformers.__version__,
            "python": platform.python_version(), "platform": platform.platform()}


def synthetic_rows(count, questions=5):
    """Long states plus both a large option set and joint multi-question RLCD."""
    if count < 1 or questions < 1 or questions > 5:
        raise ValueError("Use positive rows and one to five questions")
    definitions = {
        "route": {"type": "choice", "instructions": "Route this synthetic training example.",
                  "criteria": [f"intent {i}" for i in range(77)]},
        "relevant": {"type": "noul", "instructions": "Is this synthetic example relevant?", "criteria": {}},
        "risk": {"type": "score", "instructions": "Rate its synthetic risk.",
                 "criteria": ["very low", "low", "medium", "high", "very high"]},
        "outcome": {"type": "choice", "instructions": "Select a synthetic outcome.",
                    "criteria": ["success", "partial", "failure"]},
        "review": {"type": "noul", "instructions": "Does it need synthetic review?", "criteria": {}},
    }
    targets = {"route": {f"intent {i}": float(i == 3) for i in range(77)},
               "relevant": {"false": 0., "true": 1.},
               "risk": {str(i): float(i == 2) for i in range(5)},
               "outcome": {"success": 1., "partial": 0., "failure": 0.},
               "review": {"false": 1., "true": 0.}}
    selected = list(definitions)[:questions]
    return [{"state": f"Synthetic record {i}. " + "training context " * 4096,
             "questions": {k: definitions[k] for k in selected},
             "targets": {k: targets[k] for k in selected}} for i in range(count)]


def validate_pilot_config(config):
    from .data import dataset_configs
    model, training = config["model"], config["training"]
    if (config["device"] != "cuda" or not model.get("train_encoder") or
            model.get("encoder_last_n_layers") is not None or model["max_len"] != 1024 or
            model.get("mixed_precision") != "bf16" or not model.get("norm_first") or
            not model.get("independent_init") or not model.get("gradient_checkpointing") or
            config["ablation"].get("no_transformer") or config["ablation"].get("linear_scorer") or
            config["ablation"]["objective"] != "sampled_reward" or
            config["ablation"]["reward_weights"] != {"log": 1., "spherical": .5, "rps": 1.}):
        raise ValueError("Pilot requires complete encoder, pre-norm independent transformer, BF16, 1024 tokens and full RLCD")
    datasets = dataset_configs(config)
    if any(d["name"] in {"ms_marco", "commitpackft"} for d in datasets):
        raise ValueError("MS MARCO and CommitPackFT are excluded")
    if not config["data"].get("separate_validation") or training.get("validation_role") != "validation":
        raise ValueError("Pilot checkpoint selection must use a separate validation partition")
    if not config["data"].get("prepare_train_cache") or config["data"]["train_pool_rows"] < 14:
        raise ValueError("Use bounded streaming pools with at least 14 rows for stratified DBpedia")
    return datasets
