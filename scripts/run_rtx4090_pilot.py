"""Check, preflight and run the bounded BF16 full-encoder transformer pilot on CUDA."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from decisions.checkpoints import read_checkpoint, resume_checkpoint, save_checkpoint
from decisions.cuda_pilot import check_environment, synthetic_rows, validate_pilot_config
from decisions.data import cached_training_rows
from decisions.evaluation import preprocessing
from decisions.logging import Logger
from decisions.model import load_model
from decisions.precision import build_scaler, optimizer_step
from decisions.schema import collate
from decisions.trainer import backward_microbatches, build_optimizer, memory, train


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def worker_lock(directory):
    import fcntl
    with (directory / ".worker.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pilot worker is using this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def preflight(config, directory, environment, steps=4):
    """Real optimizer/gradients, full context, joint reward and save/resume on synthetic rows."""
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device("cuda:0")
    model, tokenizer = load_model(config["model"], config["ablation"], device)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.bert.parameters()):
        raise RuntimeError("Preflight requires every encoder parameter trainable in FP32")
    optimizer = build_optimizer(model, config["training"])
    scaler = build_scaler(config["model"], device)
    optimizer._decision_scaler = scaler
    rows = synthetic_rows(config["training"]["batch_size"])
    inputs, _, _ = collate(rows, tokenizer, **preprocessing(config["model"]))
    if inputs["input_ids"].shape[1] != 1024 or not (inputs["attention_mask"].sum(1) == 1024).all():
        raise RuntimeError("Synthetic preflight must exercise 1024 actual tokens per question")
    question_count = len(inputs["qtype"])
    del inputs
    watched = {k: p for k, p in model.bert.named_parameters()
               if k in {"layers.0.attn.Wqkv.weight", "layers.27.mlp.Wi.weight", "final_norm.weight"}}
    if len(watched) != 3:
        raise RuntimeError("Pinned ModernBERT early/late parameter audit unavailable")
    initial = {k: p.detach().flatten()[:16].cpu().clone() for k, p in watched.items()}
    logger = Logger(config["logging"], directory.with_name(directory.name + "__preflight"),
                    {**config, "protocol": "Synthetic CUDA preflight; no dataset or evaluation scores"})
    records = []
    try:
        for step in range(steps):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            microbatches = [rows] * config["training"]["gradient_accumulation_steps"]
            loss = backward_microbatches(model, tokenizer, microbatches, device, config, scaler)
            if not all(p.grad is not None and p.grad.dtype == torch.float32 and
                       torch.isfinite(p.grad).all() for p in watched.values()):
                raise RuntimeError("Early/late encoder gradients must be finite FP32")
            norm, updated = optimizer_step(optimizer, parameters, config["training"]["clip_grad"], scaler)
            if not updated or norm is None:
                raise RuntimeError("Synthetic optimizer update was skipped")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            record = {"step": step + 1, "preflight/loss": loss, "preflight/gradient_norm": norm,
                      "preflight/seconds": time.perf_counter() - started, **memory(device)}
            records.append(record)
            logger.log(record, step + 1)
            print(json.dumps(record), flush=True)
        changes = {k: (p.detach().flatten()[:16].cpu() - initial[k]).abs().max().item()
                   for k, p in watched.items()}
        if not all(v > 0 for v in changes.values()):
            raise RuntimeError("Early/late encoder weight update audit failed")
        if not all(v.dtype == torch.float32 for state in optimizer.state.values()
                   for k, v in state.items() if k in {"exp_avg", "exp_avg_sq"}):
            raise RuntimeError("Adam moments must remain FP32")
        checkpoint_config = {**config, "dataset_configs": []}
        progress = {"step": steps, "rows": steps * len(rows) * config["training"]["gradient_accumulation_steps"]}
        with tempfile.TemporaryDirectory(prefix="checkpoint-preflight-", dir=directory) as temporary:
            path = Path(temporary) / "last.pt"
            save_checkpoint(path, model, optimizer, checkpoint_config, progress, device)
            expected_rng = torch.rand(16, device=device)
            key = next(iter(watched))
            expected_weight = watched[key].detach().flatten()[:16].clone()
            with torch.no_grad():
                watched[key].add_(.125)
            restored = resume_checkpoint(path, model, optimizer, checkpoint_config, [], device)
            torch.testing.assert_close(torch.rand(16, device=device), expected_rng, rtol=0, atol=0)
            torch.testing.assert_close(watched[key].detach().flatten()[:16], expected_weight, rtol=0, atol=0)
            if restored != progress:
                raise RuntimeError("Checkpoint progress recovery failed")
        return {"status": "passed", "protocol": "Synthetic CUDA full-length multi-question preflight; no dataset scores",
                "environment": environment, "model": config["model"], "training": config["training"],
                "questions_per_microbatch_row": 5, "collated_questions": question_count,
                "question_forward_microbatch_size": config["training"]["question_microbatch_size"],
                "trainable_parameters": sum(p.numel() for p in parameters), "steps": records,
                "encoder_parameter_sample_max_changes": changes, "checkpoint_rng_resume": "passed",
                "mean_update_seconds_excluding_first": statistics.mean(r["preflight/seconds"] for r in records[1:])}
    finally:
        logger.finish()
        del model, tokenizer, parameters, optimizer, watched, scaler
        gc.collect()
        torch.cuda.empty_cache()


def input_identity(row):
    return hashlib.sha256(json.dumps({"state": row["state"], "questions": row["questions"]},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(directory, config, datasets, environment):
    saved = read_checkpoint(directory / "last.pt")
    progress = saved["progress"]
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if not metrics or not progress["rows"]:
        raise RuntimeError("No optimizer updates completed; inspect logs before using validation scores")
    history = [json.loads(line) for line in (directory / "validation_history.jsonl").read_text().splitlines()]
    probes = [json.loads(line) for line in (directory / "feature_diagnostics.jsonl").read_text().splitlines()]
    pool_counts, hashes, overlap = {}, {}, {}
    for dataset in datasets:
        name = dataset["name"]
        source = directory / "training_samples" / f"{name}.jsonl"
        training_ids = [input_identity(row) for row in cached_training_rows(source)]
        pool_counts[name] = len(training_ids)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        validation_path = next((directory / "validation_samples").glob(f"{name}-*.json"))
        validation_rows = json.loads(validation_path.read_text())["rows"]
        overlap[name] = len(set(training_ids) & {input_identity(row) for row in validation_rows})
    if any(overlap.values()):
        raise RuntimeError("Training/validation input overlap detected; do not use these scores")
    summary = {"protocol": "Bounded CUDA transformer pilot; validation-only selection; no final evaluation",
               "environment": environment, "config": config, "progress": progress,
               "training_pool_counts": pool_counts, "training_pool_sha256": hashes,
               "train_validation_input_overlap": overlap, "validation_history": history,
               "selected_validation": next(r for r in history if r["step"] == progress["best_step"]),
               "feature_diagnostics": probes,
               "successful_optimizer_updates": sum(m["train/optimizer_updated"] for m in metrics),
               "peak_logged_cuda_allocated_gib": max(m["memory/peak_allocated_gib"] for m in metrics),
               "peak_logged_cuda_reserved_gib": max(m["memory/peak_reserved_gib"] for m in metrics),
               "mean_training_seconds_per_row": sum(m["train/seconds"] for m in metrics) / progress["rows"],
               "timing_scope": "Synchronized update timing includes input loading/collation; periodic validation/checkpointing excluded",
               "limitations": ["Bounded source pools, one seed; not a full-corpus run or official Decision Index score.",
                               "New sources, BF16 and LR warmup differ from the M1 smoke; this is not an isolated hardware ablation.",
                               "CUDA peaks cover training update windows; diagnostics and validation run outside them."]}
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/rtx4090_pilot/prenorm_bf16_b4")
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--question-microbatch-size", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--plan", action="store_true", help="Resolve config without GPU, data or model access")
    parser.add_argument("--check-env", action="store_true", help="Check CUDA/hardware/disk without downloading model/data")
    parser.add_argument("--preflight-only", action="store_true", help="Synthetic GPU training and checkpoint recovery only")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt with identical settings and accumulated budget")
    args = parser.parse_args()
    if args.resume and args.preflight_only:
        parser.error("Use --preflight-only without --resume and with a fresh directory")
    directory = args.run_dir.resolve()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=rtx4090_pilot"])
    cfg.training.batch_size = args.batch_size
    cfg.training.gradient_accumulation_steps = 4 // args.batch_size
    cfg.training.question_microbatch_size = args.question_microbatch_size
    # Stable absolute cache path is part of exact checkpoint provenance.
    cfg.data.train_cache_dir = str(directory / "training_samples")
    if args.resume:
        cfg.resume = str(directory / "last.pt")
    config = OmegaConf.to_container(cfg, resolve=True)
    datasets = validate_pilot_config(config)
    if args.plan:
        print(json.dumps({"run_dir": str(directory), "source_count": len(datasets),
                          "maximum_training_pool_rows": len(datasets) * config["data"]["train_pool_rows"],
                          "config": config}, indent=2))
        return
    if args.resume and not (directory / "last.pt").is_file():
        parser.error("Resume requires last.pt in the same run directory")
    if (directory / "last.pt").exists() and not args.resume and not args.check_env:
        parser.error("Existing training checkpoint: use --resume or a new --run-dir")
    directory.mkdir(parents=True, exist_ok=True)
    with worker_lock(directory):
        def status(phase, **values):
            write_json(directory / "status.json", {"phase": phase,
                       "updated_at": datetime.now(timezone.utc).isoformat(), **values})
        try:
            environment = check_environment(config, directory)
            write_json(directory / "environment.json", environment)
            if args.check_env:
                print(json.dumps(environment, indent=2))
                return
            if not args.resume:
                status("preflight")
                result = preflight(config, directory, environment)
                write_json(directory / "preflight.json", result)
            if args.preflight_only:
                status("preflight_passed")
                return
            status("training", protocol="Bounded streaming pools; validation only")
            train(cfg, directory)
            status("summarizing")
            summary = summarize(directory, config, datasets, environment)
            status("completed", rows=summary["progress"]["rows"], steps=summary["progress"]["step"],
                   summary=str(directory / "summary.json"))
            print("Pilot completed: " + str(directory / "summary.json"), flush=True)
        except BaseException as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
