"""Resumable full-stream training, ordered by published source size."""
from __future__ import annotations

import itertools
import json
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .checkpoints import read_checkpoint, resume_checkpoint, save_checkpoint
from .cuda_pilot import configure_cuda_budget
from .data import batches, dataset_configs, examples, sampled_examples
from .logging import Logger
from .model import load_model, select_device
from .precision import build_scaler, optimizer_step
from .trainer import backward_microbatches, build_optimizer, clear_cache, memory, scheduled_learning_rate, validate


# Published raw training rows, before filters and reserved validation. Most
# counts are captured in reports/training_time_estimate.json. CodeReviewer is
# the paper's approximately 266k quality-estimation rows; FlakeFlagger is the
# paper's 26,765 tests; TREC has 5,452 published training questions. The
# customer-support count is the unfiltered release, so its actual stage is
# smaller after English filtering.
SOURCE_TRAIN_ROWS = {
    "arc_challenge": 1119,
    "typed_decisions_all": 1200,
    "openbookqa": 4957,
    "trec": 5452,
    "sst5": 8544,
    "boolq": 9427,
    "commonsenseqa": 9741,
    "banking77": 10003,
    "phishing_email": 18650,
    "imdb": 25000,
    "flakeflagger": 26765,
    "aegis": 30007,
    "enron_spam": 31716,
    "customer_support": 61765,
    "ag_news": 120000,
    "amazon_reviews_multi_en": 200000,
    "codereviewer": 266000,
    "mnli": 392702,
    "dbpedia14": 560000,
    "yelp_review_full": 650000,
    "consumer_finance": 7179332,
}


def ordered_sources(names):
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("Curriculum sources must be unique")
    missing = set(names) - SOURCE_TRAIN_ROWS.keys()
    if missing:
        raise ValueError(f"Missing published training sizes: {sorted(missing)}")
    return sorted(names, key=lambda name: (SOURCE_TRAIN_ROWS[name], name))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def replay_buffer(dataset, run_dir, settings, seed):
    """Keep a bounded training-only sample for later stage rehearsal."""
    path = Path(run_dir) / "replay" / f"{dataset['name']}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["dataset"] != dataset or saved["seed"] != seed or saved["limit"] != settings["replay_rows_per_source"]:
            raise ValueError(f"Replay provenance changed for {dataset['name']}")
        return saved["rows"]
    rows = list(sampled_examples(dataset, "train", settings["replay_rows_per_source"],
                                 seed=seed, shuffle_buffer=settings["shuffle_buffer"]))
    if not rows:
        raise ValueError(f"No training rows available for replay: {dataset['name']}")
    _write_json(path, {"dataset": dataset, "seed": seed,
                       "limit": settings["replay_rows_per_source"], "rows": rows})
    print(f"Prepared {len(rows)} training-only replay rows: {dataset['name']}", flush=True)
    return rows


def with_replay(current_rows, buffers, fraction):
    """Append balanced earlier-source rows without changing source progress."""
    if not buffers or not fraction:
        return current_rows, 0
    count = round(len(current_rows) * fraction / (1 - fraction))
    names = list(buffers)
    sampled_names = (random.sample(names, count) if count <= len(names) else
                     random.sample(names, len(names)) + random.choices(names, k=count - len(names)))
    replay = [random.choice(buffers[name]) for name in sampled_names]
    return current_rows + replay, len(replay)


def train_curriculum(cfg, run_dir):
    """Train complete streams and evaluate every source after each stage."""
    config = OmegaConf.to_container(cfg, resolve=True)
    settings = config["training"]
    if config["device"] != "cuda" or config["data"].get("prepare_train_cache"):
        raise ValueError("The full curriculum requires CUDA and uncached training streams")
    if settings["epochs"] != 1 or settings["batch_size"] < 1 or settings["gradient_accumulation_steps"] < 1:
        raise ValueError("The curriculum needs one pass and positive batch settings")
    if config["evaluation"]["max_rows"] < 1 or settings["checkpoint_every"] < 1:
        raise ValueError("Evaluation and checkpoint settings must be positive")
    replay_fraction = settings.get("replay_fraction", 0.)
    if not 0 <= replay_fraction < 1 or (replay_fraction and settings.get("replay_rows_per_source", 0) < 1):
        raise ValueError("Replay needs a fraction in [0, 1) and positive rows per source")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = dataset_configs(config)
    by_name = {dataset["name"]: dataset for dataset in datasets}
    order = ordered_sources(by_name)
    if len(datasets) != len(order):
        raise ValueError("Duplicate source configuration in the curriculum")
    config["dataset_configs"] = datasets
    config["training"]["curriculum_order"] = order
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "source_order.json", [
        {"stage": i + 1, "dataset": name, "published_train_rows": SOURCE_TRAIN_ROWS[name]}
        for i, name in enumerate(order)])

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = select_device("cuda")
    configure_cuda_budget(settings["cuda_memory_budget_gib"], device)
    model, tokenizer = load_model(config["model"], config["ablation"], device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, settings)
    scaler = build_scaler(config["model"], device)
    optimizer._decision_scaler = scaler
    progress = {"step": 0, "rows": 0, "stage_index": 0, "rows_in_stage": 0,
                "best_accuracy": -1.0, "training_elapsed_seconds": 0.0}
    if config.get("resume"):
        saved = read_checkpoint(config["resume"])
        old = saved["config"]
        if (old["training"].get("curriculum_order") != order or
                old["evaluation"] != config["evaluation"] or
                old["data"] != config["data"]):
            raise ValueError("Resume would change curriculum order, data or evaluation settings")
        for key in ("replay_fraction", "replay_rows_per_source", "auxiliary_cross_entropy_weight"):
            if key in old["training"] and old["training"][key] != settings[key]:
                raise ValueError(f"Resume would change training.{key}")
        progress = resume_checkpoint(config["resume"], model, optimizer, config, datasets, device)
    if not 0 <= progress["stage_index"] <= len(order):
        raise ValueError("Checkpoint stage index is outside the curriculum")
    started = time.monotonic()
    previous_elapsed = progress.get("training_elapsed_seconds", 0.0)
    replay_buffers = {}
    if replay_fraction:
        _write_json(run_dir / "status.json", {"phase": "preparing_replay", "stage_index": progress["stage_index"],
            "completed_stages": progress["stage_index"], "total_stages": len(order),
            "step": progress["step"], "rows": progress["rows"]})
        rng_state = random.getstate()
        try:
            for name in order[:progress["stage_index"]]:
                replay_buffers[name] = replay_buffer(by_name[name], run_dir, settings, config["seed"])
        finally:
            random.setstate(rng_state)
    logger = Logger(config["logging"], run_dir, config)
    model.train()
    update_in_progress = False
    paused = False

    def elapsed():
        return previous_elapsed + time.monotonic() - started

    def status(phase, **fields):
        _write_json(run_dir / "status.json", {"phase": phase, "stage_index": progress["stage_index"],
            "completed_stages": progress["stage_index"], "total_stages": len(order),
            "step": progress["step"], "rows": progress["rows"],
            "elapsed_seconds": elapsed(), **fields})

    try:
        if not config.get("resume"):
            save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
        with (run_dir / "metrics.jsonl").open("a") as metrics_file:
            for index in range(progress["stage_index"], len(order)):
                name = order[index]
                dataset = by_name[name]
                status("training", dataset=name, published_train_rows=SOURCE_TRAIN_ROWS[name])
                stream = examples(dataset, "train", seed=config["seed"], epoch=0,
                                  shuffle_buffer=settings["shuffle_buffer"])
                stream = itertools.islice(stream, progress["rows_in_stage"], None)
                current_batch_size = (max(1, round(settings["batch_size"] * (1 - replay_fraction)))
                                      if replay_buffers else settings["batch_size"])
                iterator = batches(stream, current_batch_size)
                while True:
                    if settings.get("max_seconds") and elapsed() >= settings["max_seconds"]:
                        paused = True
                        status("budget_reached", dataset=name)
                        break
                    if progress["step"] >= settings["max_steps"]:
                        paused = True
                        status("step_limit_reached", dataset=name)
                        break
                    checkpoint_size = (run_dir / "last.pt").stat().st_size / 2**30
                    needed_free_gib = checkpoint_size + settings.get("min_free_disk_gib", 0) + .5
                    if shutil.disk_usage(run_dir).free / 2**30 < needed_free_gib:
                        paused = True
                        status("disk_reserve_reached", dataset=name)
                        break
                    current_microbatches = list(itertools.islice(iterator, settings["gradient_accumulation_steps"]))
                    if not current_microbatches:
                        break
                    row_count = sum(len(rows) for rows in current_microbatches)
                    combined = [with_replay(rows, replay_buffers, replay_fraction) for rows in current_microbatches]
                    microbatches = [rows for rows, _ in combined]
                    replay_count = sum(count for _, count in combined)
                    if settings.get("synchronize_timing"):
                        torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    update_started = time.perf_counter()
                    update_in_progress = True
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = scheduled_learning_rate(settings, progress["step"]) * group.get("lr_scale", 1.0)
                    loss = backward_microbatches(model, tokenizer, microbatches, device, config, scaler)
                    norm, updated = optimizer_step(optimizer, parameters, settings["clip_grad"], scaler)
                    optimizer.zero_grad(set_to_none=True)
                    progress["step"] += 1
                    progress["rows"] += row_count
                    progress["rows_in_stage"] += row_count
                    progress["replay_rows"] = progress.get("replay_rows", 0) + replay_count
                    progress.setdefault("dataset_rows", {})[name] = progress["rows_in_stage"]
                    progress["training_elapsed_seconds"] = elapsed()
                    update_in_progress = False
                    if settings.get("synchronize_timing"):
                        torch.cuda.synchronize(device)
                    metrics = {"step": progress["step"], "stage_index": index + 1,
                               "rows": progress["rows"], "stage_rows": progress["rows_in_stage"],
                               "train/source_rows": row_count, "train/replay_rows": replay_count,
                               "train/loss": loss, "train/gradient_norm": norm,
                               "train/seconds": time.perf_counter() - update_started,
                               "train/optimizer_updated": int(updated),
                               "memory/process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                   (2**30 if sys.platform == "darwin" else 2**20), **memory(device)}
                    metrics_file.write(json.dumps(metrics) + "\n")
                    metrics_file.flush()
                    if progress["step"] % settings["log_every"] == 0:
                        logger.log(metrics, progress["step"])
                        print(json.dumps({"dataset": name, **metrics}), flush=True)
                    if progress["step"] % settings["checkpoint_every"] == 0:
                        save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
                if paused:
                    break
                status("evaluating", dataset=name)
                result = validate(model, tokenizer, datasets, device, config, progress, run_dir, optimizer, logger)
                if replay_fraction:
                    replay_buffers[name] = replay_buffer(dataset, run_dir, settings, config["seed"])
                progress["stage_index"] = index + 1
                progress["rows_in_stage"] = 0
                progress["training_elapsed_seconds"] = elapsed()
                stage_path = run_dir / "stages" / f"{index + 1:02d}-{name}.pt"
                save_checkpoint(stage_path, model, optimizer, config, progress, device, include_optimizer=False)
                save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
                _write_json(run_dir / "stages" / f"{index + 1:02d}-{name}.json", {
                    "stage": index + 1, "dataset": name, "published_train_rows": SOURCE_TRAIN_ROWS[name],
                    "trained_rows": progress["dataset_rows"][name], "step": progress["step"],
                    "replay_rows_total": progress.get("replay_rows", 0),
                    "macro_validation_accuracy": result["macro_accuracy"],
                    "validation": result["report"], "checkpoint": str(stage_path)})
                status("stage_completed", dataset=name, stage_checkpoint=str(stage_path))
                print(f"Completed stage {index + 1}/{len(order)}: {name}", flush=True)
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        if not update_in_progress:
            progress["training_elapsed_seconds"] = elapsed()
            save_checkpoint(run_dir / "last.pt", model, optimizer, config, progress, device)
        else:
            print("Update interrupted; previous atomic checkpoint retained", flush=True)
        logger.finish()
        clear_cache(device)
    if not paused:
        status("completed")
    return progress
