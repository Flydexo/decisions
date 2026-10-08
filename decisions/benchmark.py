"""Strict reference-engine adapter and an explicitly partial streaming source pilot."""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import torch

from .data import load_stream
from .evaluation import Metrics, Predictor
from .schema import Unsupported as RequestUnsupported

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:
    # Allows the small source pilot without installing the optional reproduction kit.
    class Engine:
        def __init__(self, **options):
            self.options = options

    Unsupported = RequestUnsupported


class DecisionEngine(Engine):
    name = "frozen-modernbert-decisions"
    latency = "Synchronized in-process request time, including tokenization; excludes model loading."

    def __init__(self, checkpoint, device="auto", question_batch_size=2, temperature=1.0, **options):
        super().__init__(checkpoint=checkpoint, device=device, question_batch_size=question_batch_size,
                         temperature=temperature, **options)
        self.predictor = Predictor(checkpoint, device, strict=True, question_batch_size=int(question_batch_size),
                                   temperature=float(temperature))
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as handle:
            for block in iter(lambda: handle.read(2**20), b""):
                digest.update(block)
        self.provenance = {"checkpoint_sha256": digest.hexdigest(), "encoder": self.predictor.config["model"],
                           "ablation": self.predictor.config["ablation"],
                           "training_sources": self.predictor.config["dataset_configs"],
                           "strict_context": True, "confidence": "calibrated chosen-option probability",
                           "temperature": self.predictor.temperature}

    def __call__(self, state, questions):
        try:
            response = self.predictor(state, questions)
        except RequestUnsupported as error:
            raise Unsupported(str(error)) from error
        return response, {"confidence": {key: answer["confidence"] for key, answer in response["answers"].items()}}

    def synchronize(self):
        if self.predictor.device.type == "mps":
            torch.mps.synchronize()
        elif self.predictor.device.type == "cuda":
            torch.cuda.synchronize()

    def runtime(self):
        return {"device": str(self.predictor.device), "torch": torch.__version__}

    def warmup(self):
        for _ in range(2):
            self("The color is red.", {"warmup": {"type": "choice", "instructions": "Which color is named?",
                                                 "criteria": {"red": "red", "blue": "blue"}}})
        self.synchronize()

    def close(self):
        pass


def choice_row(family, row, features, spec, index):
    """Mirror the upstream direct-source request layout (A/B or option_N keys)."""
    style = spec["layout"]
    if style == "arc":
        instructions, options = row["question"], row["choices"]["text"]
        gold = row["choices"]["label"].index(row["answerKey"])
    elif style == "mmlu":
        instructions, options, gold = row["question"], row["choices"], int(row["answer"])
    elif style == "winogrande":
        instructions = "Which option correctly fills the blank?\n" + row["sentence"]
        options, gold = [row["option1"], row["option2"]], int(row["answer"]) - 1
    elif style == "hellaswag":
        clean = lambda s: " ".join(str(s).split())
        instructions = "Which continuation is most plausible?\n" + clean(row["ctx"])
        options, gold = [clean(x) for x in row["endings"]], int(row["label"])
    elif style == "anli":
        instructions = ("Classify the relationship between the premise and hypothesis.\nPremise: "
                        + row["premise"] + "\nHypothesis: " + row["hypothesis"])
        options, gold = ["entailment", "neutral", "contradiction"], int(row["label"])
    elif style == "banking77":
        instructions = "Classify the banking intent of this user request:\n" + row["text"]
        options = spec.get("option_names") or features["label"].names
        gold = options.index(features["label"].names[row["label"]])
    else:
        raise ValueError(f"Unknown benchmark layout {style}")
    if not 2 <= len(options) <= 255 or not 0 <= gold < len(options):
        raise ValueError("Invalid benchmark options/target")
    keys = [chr(65 + i) if len(options) <= 26 else f"option_{i}" for i in range(len(options))]
    return {"id": f"{family}:{spec['split']}:{row.get('id', row.get('uid', index))}",
            "state": {}, "questions": {"q1": {"type": "choice", "instructions": instructions,
                                                 "criteria": dict(zip(keys, options))}},
            "expected": keys[gold]}


def source_pilot(config: dict, run_dir):
    if config["max_rows_per_source"] < 1:
        raise ValueError("max_rows_per_source must be positive")
    run_dir = Path(run_dir)
    engine = DecisionEngine(config["checkpoint"], config["device"], config["question_batch_size"])
    engine.warmup()
    report = {"protocol": "bounded streaming source-split pilot", "edition_reference": "0.2.1",
              "official_decision_index": None, "complete": False,
              "note": "First N source-split rows, not the frozen edition's selected rows. "
                      "Accuracy is over supported requests; unsupported requests are counted separately. "
                      "Entropy confidence is distinct from the chosen-option probability used for calibration.",
              "max_rows_per_source": config["max_rows_per_source"], "sources": {},
              "provenance": engine.provenance, "runtime": engine.runtime()}
    with (run_dir / "results.jsonl").open("w") as output:
        for spec in config["sources"]:
            name, metrics, unsupported, errors, total, times = spec["name"], Metrics(), 0, 0, 0, []
            source_error = None
            try:
                stream = load_stream({"source": spec["source"], "name": name}, spec["split"])
                for index, row in enumerate(itertools.islice(stream, config["max_rows_per_source"])):
                    request = choice_row(name, row, stream.features, spec, index)
                    total += 1
                    engine.synchronize()
                    started = time.perf_counter()
                    record = {"id": request["id"], "family": name, "expected": request["expected"],
                              "payload": {"state": request["state"], "questions": request["questions"]}}
                    try:
                        response, raw = engine(request["state"], request["questions"])
                        engine.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        times.append(elapsed)
                        answer = response["answers"]["q1"]
                        keys = list(request["questions"]["q1"]["criteria"])
                        p = torch.tensor([[answer["probabilities"][key] for key in keys]])
                        target = torch.tensor([[float(key == request["expected"]) for key in keys]])
                        metrics.update(p, target, torch.ones_like(p, dtype=torch.bool))
                        record.update(status="ok", response=response, raw_output=raw, total_wall_ms=elapsed)
                    except Unsupported as error:
                        unsupported += 1
                        record.update(status="unsupported", error=str(error))
                    except Exception as error:
                        errors += 1
                        record.update(status="error", error=f"{type(error).__name__}: {error}")
                    output.write(json.dumps(record) + "\n")
                    output.flush()
            except Exception as error:
                source_error = f"{type(error).__name__}: {error}"
            times.sort()
            values = {"requested": total, "answered": metrics.count, "unsupported": unsupported,
                      "errors": errors, "source_error": source_error,
                      "median_ms": times[len(times) // 2] if times else None,
                      "source": spec["source"], "split": spec["split"], **metrics.result()}
            report["sources"][name] = values
            (run_dir / "summary.json").write_text(json.dumps(report, indent=2))
            print(name + ": " + json.dumps(values), flush=True)
    engine.close()
    return report
