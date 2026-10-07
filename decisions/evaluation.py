from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path

import torch

from .data import batches, sampled_examples
from .losses import confidence
from .model import inference_logits
from .schema import Unsupported, collate, prepare_request, to_device


class Metrics:
    def __init__(self, bins=10):
        self.count, self.correct, self.nll, self.brier, self.entropy_confidence = 0, 0., 0., 0., 0.
        self.bins = bins
        self.histograms = {name: [[0, 0., 0.] for _ in range(bins)] for name in ("probability", "entropy")}

    def update(self, p, target, mask):
        p, target, mask = p.detach().cpu(), target.detach().cpu(), mask.detach().cpu()
        correct = (p.argmax(-1) == target.argmax(-1)).float()
        entropy_confidence = confidence(p, mask)
        chosen_probability = p.max(-1).values
        self.count += len(p)
        self.correct += correct.sum().item()
        self.nll += -(target * p.clamp_min(1e-12).log()).sum().item()
        self.brier += ((p - target).square() * mask).sum().item()
        self.entropy_confidence += entropy_confidence.sum().item()
        for name, values in (("probability", chosen_probability), ("entropy", entropy_confidence)):
            for value, success in zip(values.tolist(), correct.tolist()):
                cell = self.histograms[name][min(int(value * self.bins), self.bins - 1)]
                cell[0] += 1
                cell[1] += value
                cell[2] += success

    def result(self):
        if not self.count:
            return {"questions": 0, "accuracy": None}
        result = {"questions": self.count, "accuracy": self.correct / self.count,
                  "nll": self.nll / self.count, "brier": self.brier / self.count,
                  "entropy_confidence": self.entropy_confidence / self.count}
        for name, cells in self.histograms.items():
            key = "probability_ece" if name == "probability" else "entropy_accuracy_gap"
            result[key] = sum(abs(total - successes) for n, total, successes in cells if n) / self.count
        return result


def preprocessing(config, strict=False):
    return {"max_len": config["max_len"], "head_max_len": config["head_max_len"],
            "option_max_len": config["option_max_len"], "strict": strict,
            "preserve_options": config.get("preserve_options", False),
            "balanced_state": config.get("balanced_state", False)}


def evaluation_samples(dataset, config, seed):
    role = config.get("role", "eval")
    stream = lambda: sampled_examples(dataset, role, config["max_rows"], seed=seed,
                          shuffle_buffer=config.get("shuffle_buffer", 0))
    if not config.get("cache_dir"):
        return stream()
    # Only a bounded validation sample is cached, never the training dataset.
    identity = {"dataset": dataset, "role": role, "seed": seed,
                "max_rows": config["max_rows"], "shuffle_buffer": config.get("shuffle_buffer", 0)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(config["cache_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dataset['name']}-{digest[:16]}.json"
    if not path.exists():
        rows = list(stream())
        if not rows:
            raise ValueError(f"{dataset['name']}: empty {role} partition")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"provenance": identity, "rows": rows}, ensure_ascii=False))
        temporary.replace(path)
    return iter(json.loads(path.read_text())["rows"])


def evaluate(model, tokenizer, configs, device, config, seed=42):
    if config["max_rows"] < 1 or config["batch_size"] < 1:
        raise ValueError("Evaluation max_rows and batch_size must be positive")
    was_training = model.training
    model.eval()
    report = {}
    try:
        with torch.no_grad():
            for dataset in configs:
                metrics, row_count, unsupported = Metrics(), 0, 0
                stream = evaluation_samples(dataset, config, seed)
                for batch in batches(stream, config["batch_size"]):
                    supported = []
                    for row in batch:
                        try:
                            prepare_request(row, tokenizer, **preprocessing(config["preprocessing"], config["strict"]))
                            supported.append(row)
                        except Unsupported:
                            unsupported += 1
                    row_count += len(batch)
                    if supported:
                        inputs, target, _ = collate(supported, tokenizer,
                                                   **preprocessing(config["preprocessing"], config["strict"]))
                        logits = inference_logits(model, to_device(inputs, device), config.get("question_microbatch_size", 0))
                        metrics.update(logits.float().softmax(-1), target, inputs["marker_mask"])
                        del logits
                report[dataset["name"]] = {"rows": row_count, "unsupported_rows": unsupported, **metrics.result()}
    finally:
        model.train(was_training)
    return report


class Predictor:
    def __init__(self, checkpoint, device="auto", strict=True, question_batch_size=2):
        from .checkpoints import read_checkpoint, load_checkpoint_model
        from .model import load_model, select_device
        saved = read_checkpoint(checkpoint)
        self.config = saved["config"]
        self.device = select_device(device)
        self.model, self.tokenizer = load_model(self.config["model"], self.config["ablation"], self.device)
        load_checkpoint_model(self.model, saved)
        self.model.eval()
        self.strict = strict
        self.question_batch_size = question_batch_size
        if question_batch_size < 1:
            raise ValueError("question_batch_size must be positive")

    @torch.no_grad()
    def __call__(self, state, questions):
        # Validate the complete request before executing any subset of its questions.
        inputs = prepare_request({"state": state, "questions": questions}, self.tokenizer,
                                 **preprocessing(self.config["model"], self.strict))
        answers = {}
        for start in range(0, len(inputs["qtype"]), self.question_batch_size):
            end = start + self.question_batch_size
            chunk = {k: v[start:end] for k, v in inputs.items()}
            p = self.model(to_device(chunk, self.device)).float().softmax(-1).cpu()
            certainty = confidence(p, chunk["marker_mask"])
            for index, (qid, keys) in enumerate(zip(chunk["question_ids"], chunk["option_labels"])):
                values = p[index, :len(keys)].tolist()
                selected = max(range(len(values)), key=values.__getitem__)
                kind = questions[qid]["type"]
                answer = {"type": kind, "probabilities": dict(zip(keys, values)),
                          "confidence": certainty[index].item(), "chosen_probability": values[selected]}
                if kind == "noul":
                    answer["noul"] = answer["probabilities"]["true"]
                elif kind == "score":
                    answer["score"] = sum(int(key) * value for key, value in zip(keys, values))
                else:
                    answer["choice"] = keys[selected]
                answers[qid] = answer
        return {"model": "finetuned-modernbert-decisions" if self.config["model"].get("train_encoder", False) else "frozen-modernbert-decisions",
                "answers": answers,
                "usage": {"input_tokens": int(inputs["attention_mask"].sum())}}
