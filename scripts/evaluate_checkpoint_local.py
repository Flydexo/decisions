"""Evaluate a saved curriculum checkpoint on every validation source locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from decisions.checkpoints import load_checkpoint_model, read_checkpoint
from decisions.evaluation import evaluate
from decisions.model import load_model, select_device


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--compute-precision", choices=("auto", "checkpoint", "fp32"), default="auto",
                        help="auto uses FP32 on CPU because CPU BF16 autocast can be very slow")
    parser.add_argument("--datasets", nargs="+", help="Evaluate only these named sources")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    checkpoint = args.checkpoint.resolve()
    output = (args.output or checkpoint.with_suffix(".validation.json")).resolve()
    saved = read_checkpoint(checkpoint)
    config = saved["config"]
    datasets = config["dataset_configs"]
    if args.datasets:
        unknown = set(args.datasets) - {dataset["name"] for dataset in datasets}
        if unknown:
            parser.error(f"Unknown datasets: {', '.join(sorted(unknown))}")
        datasets = [dataset for dataset in datasets if dataset["name"] in set(args.datasets)]
    names = [dataset["name"] for dataset in datasets]
    compute_precision = ("fp32" if args.device == "cpu" else config["model"].get("mixed_precision", "fp32")) \
        if args.compute_precision == "auto" else (
            config["model"].get("mixed_precision", "fp32") if args.compute_precision == "checkpoint" else "fp32")
    identity = {"checkpoint_sha256": checksum(checkpoint), "checkpoint": str(checkpoint),
                "device": args.device, "compute_precision": compute_precision,
                "validation_role": "validation", "seed": config["seed"],
                "evaluation": config["evaluation"], "datasets": names,
                "progress": saved["progress"]}
    if output.exists():
        report = json.loads(output.read_text())
        if report["identity"] != identity:
            parser.error(f"Existing report has different checkpoint or evaluation settings: {output}")
    else:
        report = {"identity": identity, "validation": {}, "errors": {}}
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    model, tokenizer = load_model(config["model"], config["ablation"], device)
    load_checkpoint_model(model, saved)
    model.precision_settings["mixed_precision"] = compute_precision
    model.eval()
    evaluation = dict(config["evaluation"], preprocessing=config["model"], role="validation",
                      cache_dir=str(output.parent / "validation_samples"))
    for dataset in datasets:
        name = dataset["name"]
        if name in report["validation"]:
            continue
        started = time.monotonic()
        try:
            result = evaluate(model, tokenizer, [dataset], device, evaluation, config["seed"])
            report["validation"].update(result)
            report["errors"].pop(name, None)
            print(f"{name}: accuracy={result[name]['accuracy']} rows={result[name]['rows']} "
                  f"seconds={time.monotonic() - started:.1f}", flush=True)
        except Exception as exc:
            report["errors"][name] = f"{type(exc).__name__}: {exc}"
            print(f"{name}: ERROR {report['errors'][name]}", flush=True)
        write_json(output, report)
    scores = [row["accuracy"] for row in report["validation"].values() if row["accuracy"] is not None]
    report["macro_validation_accuracy"] = sum(scores) / len(scores) if scores else None
    report["completed_sources"] = len(report["validation"])
    write_json(output, report)
    print(json.dumps({"output": str(output), "completed_sources": report["completed_sources"],
                      "total_sources": len(datasets), "macro_validation_accuracy": report["macro_validation_accuracy"],
                      "errors": report["errors"]}), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
