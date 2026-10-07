from __future__ import annotations

import gc
import itertools
import json
import math
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .checkpoints import resume_checkpoint, save_checkpoint
from .data import batches, dataset_configs, mixed_examples, prepare_training_cache
from .diagnostics import training_probe_rows, log_feature_probe
from .evaluation import evaluate, preprocessing
from .logging import Logger
from .losses import training_loss
from .model import load_model, select_device, training_logits
from .schema import collate, to_device
from .precision import build_scaler, optimizer_step


def memory(device):
    if device.type == "mps":
        return {"memory/live_gib": torch.mps.current_allocated_memory() / 2**30,
                "memory/driver_gib": torch.mps.driver_allocated_memory() / 2**30}
    if device.type == "cuda":
        return {"memory/live_gib": torch.cuda.memory_allocated() / 2**30,
                "memory/driver_gib": torch.cuda.memory_reserved() / 2**30}
    return {}


def clear_cache(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def scheduled_learning_rate(settings, step):
    """Deterministic update-based schedule, including across exact resume."""
    warmup = settings.get("warmup_steps", 0)
    decay = settings.get("decay_steps", 0)
    if warmup and step < warmup:
        ratio = (step + 1) / warmup
    elif decay:
        phase = min(1., max(0., (step - warmup) / max(1, decay - warmup)))
        floor = settings.get("min_learning_rate_ratio", .1)
        ratio = floor + (1 - floor) * (1 + math.cos(math.pi * phase)) / 2
    else:
        ratio = 1.
    return settings["learning_rate"] * ratio


def build_optimizer(model, settings):
    encoder = [p for p in model.bert.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    kwargs = {"lr": settings["learning_rate"], "weight_decay": settings["weight_decay"],
              "foreach": settings.get("optimizer_foreach")}
    if not encoder:
        return torch.optim.AdamW(head, **kwargs)
    encoder_lr = settings.get("encoder_learning_rate", settings["learning_rate"])
    if encoder_lr <= 0 or settings["learning_rate"] <= 0:
        raise ValueError("Encoder and head learning rates must be positive")
    return torch.optim.AdamW([
        {"params": head, "name": "head", "lr_scale": 1.},
        {"params": encoder, "name": "encoder", "lr": encoder_lr,
         "lr_scale": encoder_lr / settings["learning_rate"]}], **kwargs)


def backward_microbatches(model, tokenizer, microbatches, device, config, scaler=None):
    """Accumulate row-weighted losses; retain the joint reward within each row."""
    total_rows = sum(len(rows) for rows in microbatches)
    loss_value = 0.
    for rows in microbatches:
        inputs, target, spans = collate(rows, tokenizer, **preprocessing(config["model"]))
        inputs, target = to_device(inputs, device), target.to(device)
        logits = training_logits(model, inputs, config["training"].get("question_microbatch_size", 0))
        loss = training_loss(logits, target, inputs, spans, config["training"], config["ablation"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        weight = len(rows) / total_rows
        weighted = loss * weight
        (scaler.scale(weighted) if scaler is not None else weighted).backward()
        loss_value += loss.item() * weight
        del inputs, target, logits, loss, weighted
        if config["training"].get("clear_cache_between_microbatches", False):
            clear_cache(device)
    return loss_value


def train(cfg, run_dir):
    config = OmegaConf.to_container(cfg, resolve=True)
    settings = config["training"]
    for key in ("batch_size", "epochs", "max_steps", "log_every", "checkpoint_every", "cache_clear_every"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    accumulation = settings.get("gradient_accumulation_steps", 1)
    question_size = settings.get("question_microbatch_size", 0)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if isinstance(question_size, bool) or not isinstance(question_size, int) or question_size < 0:
        raise ValueError("question_microbatch_size must be a nonnegative integer")
    if settings["shuffle_buffer"] < 0:
        raise ValueError("shuffle_buffer must be nonnegative")
    if config["evaluation"]["max_rows"] < 1:
        raise ValueError("evaluation.max_rows must be positive")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = dataset_configs(config)
    config["dataset_configs"] = datasets
    data_settings = config.get("data", {})
    if data_settings.get("prepare_train_cache"):
        cache_dir = data_settings.get("train_cache_dir") or str(run_dir / "training_samples")
        config["data"]["train_cache_dir"] = cache_dir
        # Prepare one bounded reader at a time before encoder/Adam allocation.
        prepare_training_cache(config, datasets, Path(cache_dir))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = select_device(config["device"])
    if device.type == "mps" and settings.get("mps_memory_budget_gib"):
        fraction = settings["mps_memory_budget_gib"] * 2**30 / torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(fraction)
    model, tokenizer = load_model(config["model"], config["ablation"], device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, settings)
    scaler = build_scaler(config["model"], device)
    optimizer._decision_scaler = scaler
    progress = {"step": 0, "epoch": 0, "rows_in_epoch": 0, "rows": 0, "best_accuracy": -1.0}
    if settings.get("budget_spent_seconds"):
        progress["training_elapsed_seconds"] = settings["budget_spent_seconds"]
    if config.get("resume"):
        progress = resume_checkpoint(config["resume"], model, optimizer, config, datasets, device)
    logger = Logger(config["logging"], run_dir, config)
    diagnostic_settings = config.get("diagnostics", {})
    probe_rows = (training_probe_rows(datasets, config.get("data", {}).get("train_cache_dir"),
                                     diagnostic_settings["rows_per_dataset"])
                  if diagnostic_settings.get("every_steps") else None)
    diagnostic_step = None
    print(f"Device: {device}; trainable parameters: {sum(p.numel() for p in parameters):,}; datasets: "
          + ", ".join(d["name"] for d in datasets), flush=True)
    model.train()
    update_in_progress = False
    training_started = time.monotonic()
    elapsed_before_resume = progress.get("training_elapsed_seconds", 0.)
    stop_training = False
    try:
        save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
        if probe_rows:
            log_feature_probe(model, tokenizer, probe_rows, device, config, run_dir, logger, progress["step"])
            diagnostic_step = progress["step"]
        with (run_dir / "metrics.jsonl").open("a") as metrics_file:
            for epoch in range(progress["epoch"], settings["epochs"]):
                if progress["step"] >= settings["max_steps"] or stop_training:
                    break
                stream = mixed_examples(datasets, "train", seed=config["seed"], epoch=epoch,
                                        shuffle_buffer=settings["shuffle_buffer"],
                                        cache_dir=config.get("data", {}).get("train_cache_dir"))
                skip = progress["rows_in_epoch"] if epoch == progress["epoch"] else 0
                stream = itertools.islice(stream, skip, None)
                progress["epoch"] = epoch
                progress["rows_in_epoch"] = skip
                iterator = iter(batches(stream, settings["batch_size"]))
                while progress["step"] < settings["max_steps"]:
                    if (settings.get("min_free_disk_gib", 0) and
                        shutil.disk_usage(run_dir).free / 2**30 < settings["min_free_disk_gib"]):
                        print("Disk reserve reached; stopping before a checkpoint can exhaust storage.", flush=True)
                        stop_training = True
                        break
                    elapsed = elapsed_before_resume + time.monotonic() - training_started
                    if settings.get("max_seconds", 0) and elapsed >= settings["max_seconds"]:
                        print("Training time budget reached; selecting the best validation checkpoint.", flush=True)
                        stop_training = True
                        break
                    started = time.perf_counter()
                    microbatches = list(itertools.islice(iterator, accumulation))
                    if not microbatches:
                        progress["epoch"], progress["rows_in_epoch"] = epoch + 1, 0
                        break
                    # Retain the last atomic checkpoint if any accumulated
                    # forward/backward or optimizer update is interrupted.
                    update_in_progress = True
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = scheduled_learning_rate(settings, progress["step"]) * group.get("lr_scale", 1.)
                    loss_value = backward_microbatches(model, tokenizer, microbatches, device, config, scaler)
                    norm_value, updated = optimizer_step(optimizer, parameters, settings["clip_grad"], scaler)
                    optimizer.zero_grad(set_to_none=True)
                    rows = [row for micro in microbatches for row in micro]
                    progress["step"] += 1
                    progress["rows"] += len(rows)
                    progress["rows_in_epoch"] += len(rows)
                    if settings.get("max_seconds"):
                        progress["training_elapsed_seconds"] = elapsed_before_resume + time.monotonic() - training_started
                    counts = progress.setdefault("dataset_rows", {})
                    for row in rows:
                        name = row.get("_dataset", datasets[0]["name"])
                        counts[name] = counts.get(name, 0) + 1
                    update_in_progress = False
                    del rows, microbatches
                    step = progress["step"]
                    if step % settings["cache_clear_every"] == 0:
                        clear_cache(device)
                    metrics = {"step": step, "rows": progress["rows"], "train/loss": loss_value,
                               "train/gradient_norm": norm_value, "train/seconds": time.perf_counter() - started,
                               "train/learning_rate": optimizer.param_groups[0]["lr"],
                               "train/optimizer_updated": int(updated), "train/loss_scale": scaler.get_scale(),
                               "memory/process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20),
                               **memory(device)}
                    for group in optimizer.param_groups:
                        if group.get("name") == "encoder":
                            metrics["train/encoder_learning_rate"] = group["lr"]
                    metrics_file.write(json.dumps(metrics) + "\n")
                    metrics_file.flush()
                    if step % settings["log_every"] == 0:
                        logger.log(metrics, step)
                        print(json.dumps(metrics), flush=True)
                    if step % settings["checkpoint_every"] == 0:
                        save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
                    if probe_rows and step % diagnostic_settings["every_steps"] == 0:
                        log_feature_probe(model, tokenizer, probe_rows, device, config, run_dir, logger, step)
                        diagnostic_step = step
                    every = config["evaluation"].get("every_steps", 0)
                    if every and step % every == 0:
                        validate(model, tokenizer, datasets, device, config, progress, run_dir, optimizer, logger)
                        patience = settings.get("early_stopping_patience", 0)
                        if patience and progress.get("validation_without_improvement", 0) >= patience:
                            print("Validation stopped improving; stopping early.", flush=True)
                            stop_training = True
                            break
            if probe_rows and diagnostic_step != progress["step"]:
                log_feature_probe(model, tokenizer, probe_rows, device, config, run_dir, logger, progress["step"])
            validate(model, tokenizer, datasets, device, config, progress, run_dir, optimizer, logger)
    finally:
        # An interrupted optimizer can contain a partial update: retain the last
        # atomic checkpoint rather than replacing it with inconsistent state.
        optimizer.zero_grad(set_to_none=True)
        if settings.get("max_seconds"):
            progress["training_elapsed_seconds"] = elapsed_before_resume + time.monotonic() - training_started
        try:
            if not update_in_progress:
                save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
            else:
                print(f"Update interrupted; retained the previous checkpoint at {run_dir / 'last.pt'}", flush=True)
        finally:
            logger.finish()
    print(f"Checkpoint: {run_dir / 'last.pt'}", flush=True)
    return run_dir


def validate(model, tokenizer, datasets, device, config, progress, run_dir, optimizer, logger):
    role = config["training"].get("validation_role", "eval")
    evaluation = dict(config["evaluation"], preprocessing=config["model"], role=role)
    if role == "validation":
        evaluation["cache_dir"] = str(run_dir / "validation_samples")
    report = evaluate(model, tokenizer, datasets, device, evaluation, config["seed"])
    (run_dir / ("validation.json" if role == "validation" else "evaluation.json")).write_text(json.dumps(report, indent=2))
    scores = [r["accuracy"] for r in report.values() if r["accuracy"] is not None]
    for name, values in report.items():
        logger.log({f"{role}/{name}/{key}": value for key, value in values.items() if value is not None}, progress["step"])
    score = sum(scores) / len(scores) if scores else -1.
    improved = scores and score > progress["best_accuracy"]
    if role == "validation":
        progress["validation_without_improvement"] = 0 if improved else progress.get("validation_without_improvement", 0) + 1
        with (run_dir / "validation_history.jsonl").open("a") as history:
            history.write(json.dumps({"step": progress["step"], "rows": progress["rows"],
                                      "macro_accuracy": score, "improved": bool(improved), "datasets": report}) + "\n")
    if improved:
        progress["best_accuracy"] = score
        progress["best_step"] = progress["step"]
        save_checkpoint(run_dir / "best.pt", model, optimizer, config, progress, device,
                        include_optimizer=config["training"].get("best_checkpoint_optimizer", True))
    clear_cache(device)
    print("Evaluation: " + json.dumps(report), flush=True)
