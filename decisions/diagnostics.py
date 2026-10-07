"""Fixed training-only probes for detecting option-feature collapse."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch

from .data import batches, cached_training_rows
from .evaluation import Metrics, preprocessing
from .losses import reward
from .schema import collate, to_device


def training_probe_rows(datasets, cache_dir, rows_per_dataset):
    if not cache_dir or rows_per_dataset < 1:
        raise ValueError("Feature diagnostics need a training cache and a positive row count")
    rows = []
    for dataset in datasets:
        stream = cached_training_rows(Path(cache_dir) / f"{dataset['name']}.jsonl")
        try:
            rows.extend(itertools.islice(stream, rows_per_dataset))
        finally:
            stream.close()
    if not rows:
        raise ValueError("Empty training-only feature probe")
    return rows


def marker_statistics(features, inputs):
    positions = inputs["marker_pos"].unsqueeze(-1).expand(-1, -1, features.shape[-1])
    markers = features.gather(1, positions)
    statistics = []
    for values, valid in zip(markers, inputs["marker_mask"]):
        values = values[valid]
        centered = values - values.mean(0, keepdim=True)
        unit = torch.nn.functional.normalize(values, dim=-1)
        pairs = unit @ unit.T
        off_diagonal = ~torch.eye(len(values), device=values.device, dtype=torch.bool)
        statistics.append({"marker_rms": centered.square().mean().sqrt().item(),
                           "marker_cosine": pairs[off_diagonal].mean().item() if len(values) > 1 else 1.,
                           "marker_norm": values.norm(dim=-1).mean().item()})
    return statistics


@torch.no_grad()
def feature_probe(model, tokenizer, rows, device, config, batch_size=2):
    was_training = model.training
    model.eval()
    collected, scoring = {}, Metrics()
    rewards = {key: [] for key in ("log", "spherical", "rps")}
    try:
        for batch in batches(rows, batch_size):
            inputs, target, spans = collate(batch, tokenizer, **preprocessing(config["model"]))
            inputs, target = to_device(inputs, device), target.to(device)
            features = model.encode(inputs)
            stages = {"encoder": marker_statistics(features, inputs)}
            if model.question_type is not None:
                features = features + model.question_type(inputs["qtype"]).unsqueeze(1)
                stages["question_type"] = marker_statistics(features, inputs)
            if model.transformer is not None:
                for index, layer in enumerate(model.transformer.layers, 1):
                    features = layer(features, src_key_padding_mask=inputs["attention_mask"] == 0)
                    stages[f"layer_{index}"] = marker_statistics(features, inputs)
                if model.transformer.norm is not None:
                    features = model.transformer.norm(features)
            stages["head"] = marker_statistics(features, inputs)
            for stage, values in stages.items():
                for row in values:
                    for name, value in row.items():
                        collected.setdefault(f"features/{stage}/{name}", []).append(value)
            logits = model.score_features(features, inputs)
            p = logits.softmax(-1)
            scoring.update(p, target, inputs["marker_mask"])
            for value, valid in zip(logits, inputs["marker_mask"]):
                collected.setdefault("probe/logit_range", []).append((value[valid].max() - value[valid].min()).item())
            for start, end, width in spans:
                keys = inputs["option_labels"][start:end]
                orders = []
                for k in keys:
                    positions = (sorted(range(len(k)), key=lambda i: int(k[i]))
                                 if all(v.isdecimal() for v in k) else list(range(len(k))))
                    orders.append(positions + list(range(len(k), width)))
                order = torch.tensor(orders, device=device)
                for component in rewards:
                    weights = {name: float(name == component) for name in rewards}
                    value = reward(p[start:end, :width].unsqueeze(0), target[start:end, :width].unsqueeze(0),
                                   inputs["qtype"][start:end], inputs["marker_mask"][start:end, :width],
                                   weights, order).item()
                    rewards[component].append(-value if component == "rps" else value)
            del features, logits, p, target, inputs
    finally:
        model.train(was_training)
    result = {key: sum(values) / len(values) for key, values in collected.items()}
    result.update({f"probe/{key}": value for key, value in scoring.result().items()})
    result["features/dispersion_retention"] = result["features/head/marker_rms"] / max(
        result["features/encoder/marker_rms"], 1e-12)
    for name, values in rewards.items():
        result[f"probe/reward_{name}"] = sum(values) / len(values)
    weights = config["ablation"]["reward_weights"]
    result["probe/rlcd_reward"] = (weights["log"] * result["probe/reward_log"] +
                                   weights["spherical"] * result["probe/reward_spherical"] -
                                   weights["rps"] * result["probe/reward_rps"])
    return result


def log_feature_probe(model, tokenizer, rows, device, config, run_dir, logger, step):
    metrics = feature_probe(model, tokenizer, rows, device, config, config["diagnostics"]["batch_size"])
    with (Path(run_dir) / "feature_diagnostics.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": step, **metrics}) + "\n")
    logger.log(metrics, step)
    print("Feature diagnostic: " + json.dumps({"step": step, **metrics}), flush=True)
