"""Fit a single temperature on validation logits and assess on separate eval splits.

Raw per-question logits are cached under outputs/ so a calibration can be
reproduced without another encoder pass. Only aggregate metrics are published.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from decisions.checkpoints import load_checkpoint_model, read_checkpoint
from decisions.data import batches, sampled_examples
from decisions.evaluation import preprocessing
from decisions.model import inference_logits, load_model, select_device
from decisions.schema import Unsupported, collate, prepare_request, to_device

RUN = ROOT / "outputs/rtx4090_curriculum/full_split_ordered"
CHECKPOINT = RUN / "last_bf16_fresh_optimizer.pt"
OUT = RUN / "local_evaluation/calibration"
REPORT = ROOT / "reports/bad-laya-calibration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cached_validation(name: str) -> list[dict]:
    matches = glob.glob(str(RUN / "local_evaluation/validation_samples" / f"{name}-*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cached validation set for {name}: {matches}")
    return json.loads(Path(matches[0]).read_text())["rows"]


def collected_eval(dataset: dict, max_rows: int, seed: int) -> list[dict]:
    path = OUT / "eval_samples" / f"{dataset['name']}.json"
    identity = {"dataset": dataset, "role": "eval", "seed": seed, "max_rows": max_rows,
                "shuffle_buffer": 2048}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["identity"] != identity:
            raise ValueError(f"Stale evaluation sample cache: {path}")
        return payload["rows"]
    rows = list(sampled_examples(dataset, "eval", max_rows, seed=seed, shuffle_buffer=2048))
    if len(rows) != max_rows:
        raise ValueError(f"{dataset['name']}: expected {max_rows} eval rows; got {len(rows)}")
    save_json(path, {"identity": identity, "rows": rows})
    return rows


def extract(model, tokenizer, datasets: list[dict], config: dict, device: torch.device,
            role: str, max_rows: int) -> list[dict]:
    settings = preprocessing(config["model"], config["evaluation"]["strict"])
    path = OUT / f"{role}-logits.json"
    identity = {"checkpoint_sha256": sha256(CHECKPOINT), "role": role,
                "max_rows": max_rows, "seed": config["seed"], "strict": config["evaluation"]["strict"],
                "datasets": [dataset["name"] for dataset in datasets]}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["identity"] != identity:
            raise ValueError(f"Stale logits cache: {path}")
        return payload["questions"]
    result = []
    with torch.inference_mode():
        for dataset in datasets:
            name = dataset["name"]
            rows = cached_validation(name) if role == "validation" else collected_eval(
                dataset, max_rows, config["seed"])
            unsupported = 0
            for batch in batches(rows, config["evaluation"]["batch_size"]):
                supported = []
                for row in batch:
                    try:
                        prepare_request(row, tokenizer, **settings)
                        supported.append(row)
                    except Unsupported:
                        unsupported += 1
                if not supported:
                    continue
                inputs, target, spans = collate(supported, tokenizer, **settings)
                logits = inference_logits(model, to_device(inputs, device),
                                          config["evaluation"].get("question_microbatch_size", 0))
                logits = logits.float().cpu()
                for row_index, (start, end, _) in enumerate(spans):
                    for index in range(start, end):
                        width = len(inputs["option_labels"][index])
                        result.append({"dataset": name, "row_id": str(supported[row_index].get("id", "")),
                                       "question_id": inputs["question_ids"][index],
                                       "logits": logits[index, :width].tolist(),
                                       "target": target[index, :width].tolist()})
            print(f"{role} {name}: {sum(q['dataset'] == name for q in result)} questions, "
                  f"{unsupported} unsupported rows", flush=True)
    save_json(path, {"identity": identity, "questions": result})
    return result


def tensorize(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["logits"]) for row in rows)
    logits = torch.full((len(rows), width), -torch.inf, dtype=torch.float64)
    target = torch.zeros_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, row in enumerate(rows):
        size = len(row["logits"])
        logits[i, :size] = torch.tensor(row["logits"], dtype=torch.float64)
        target[i, :size] = torch.tensor(row["target"], dtype=torch.float64)
        mask[i, :size] = True
    if not torch.isfinite(logits[mask]).all() or not torch.allclose(target.sum(-1),
                                                                       torch.ones(len(rows), dtype=torch.float64)):
        raise ValueError("Nonfinite logits or invalid targets")
    return logits, target, mask


def nll(logits: torch.Tensor, target: torch.Tensor, temperature: float) -> float:
    logp = torch.log_softmax(logits / temperature, -1)
    return -torch.where(target > 0, target * logp, 0).sum(-1).mean().item()


def fit_temperature(rows: list[dict]) -> float:
    logits, target, _ = tensorize(rows)
    # Golden section in log-space keeps T positive and covers severe overconfidence.
    left, right = math.log(0.05), math.log(1000.)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = nll(logits, target, math.exp(c)), nll(logits, target, math.exp(d))
    for _ in range(90):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = nll(logits, target, math.exp(c))
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = nll(logits, target, math.exp(d))
    temperature = math.exp((left + right) / 2)
    if not 0.051 < temperature < 999:
        raise ValueError("Optimum reached temperature search boundary")
    return temperature


def metrics(rows: list[dict], temperature: float) -> dict:
    logits, target, mask = tensorize(rows)
    p = torch.softmax(logits / temperature, -1)
    chosen = p.max(-1).values
    correct = (p.argmax(-1) == target.argmax(-1)).double()
    ece = 0.
    bins = []
    for index in range(10):
        selected = (chosen >= index / 10) & (chosen < (index + 1) / 10 if index < 9 else chosen <= 1)
        count = int(selected.sum())
        if count:
            mean_p = chosen[selected].mean().item()
            accuracy = correct[selected].mean().item()
            ece += count / len(rows) * abs(mean_p - accuracy)
            bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                         "count": count, "confidence": mean_p, "accuracy": accuracy})
    entropy = -(torch.where(mask, p * p.clamp_min(1e-300).log(), 0).sum(-1)
                / mask.sum(-1).double().log().clamp_min(1e-12))
    return {"questions": len(rows), "accuracy": correct.mean().item(),
            "mean_chosen_probability": chosen.mean().item(), "nll": nll(logits, target, temperature),
            "brier": ((p - target).square() * mask).sum(-1).mean().item(),
            "probability_ece": ece, "entropy_confidence": (1 - entropy).mean().item(),
            "bins": bins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=32)
    args = parser.parse_args()
    if args.max_rows < 1 or args.cpu_threads < 1:
        parser.error("Rows and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    saved = read_checkpoint(CHECKPOINT)
    config = saved["config"]
    if args.max_rows != config["evaluation"]["max_rows"]:
        parser.error("--max-rows must match the checkpoint's cached validation sample size")
    datasets = sorted(config["dataset_configs"], key=lambda d: d["name"])
    device = select_device(args.device)
    model, tokenizer = load_model(config["model"], config["ablation"], device)
    load_checkpoint_model(model, saved)
    model.precision_settings["mixed_precision"] = "fp32" if device.type == "cpu" else config["model"]["mixed_precision"]
    model.eval()
    fit_rows = extract(model, tokenizer, datasets, config, device, "validation", args.max_rows)
    test_rows = extract(model, tokenizer, datasets, config, device, "eval", args.max_rows)
    temperature = fit_temperature(fit_rows)
    report = {"checkpoint_sha256": sha256(CHECKPOINT), "temperature": temperature,
              "fit": {"role": "validation", "description": "Existing checkpoint-selection validation sample",
                      "raw": metrics(fit_rows, 1.), "calibrated": metrics(fit_rows, temperature)},
              "test": {"role": "eval", "description": "Separate dataset eval partitions; no temperature selection",
                       "raw": metrics(test_rows, 1.), "calibrated": metrics(test_rows, temperature)},
              "per_dataset_test": {}, "notes": ["Temperature minimizes pooled validation NLL across all sources.",
                "The validation questions were previously used to select and inspect the checkpoint.",
                "Eval questions were held out from training and temperature fitting.",
                "Temperature preserves argmax decisions; accuracy is unchanged."]}
    for dataset in datasets:
        rows = [row for row in test_rows if row["dataset"] == dataset["name"]]
        report["per_dataset_test"][dataset["name"]] = {"raw": metrics(rows, 1.),
                                                         "calibrated": metrics(rows, temperature)}
    save_json(REPORT, report)
    print(json.dumps({"report": str(REPORT), "temperature": temperature,
                      "test_raw": report["test"]["raw"], "test_calibrated": report["test"]["calibrated"]}))


if __name__ == "__main__":
    main()
