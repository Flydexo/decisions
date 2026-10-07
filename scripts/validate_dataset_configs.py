"""Check bounded training samples for the additional streaming dataset configs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from decisions.adapters import Adapter
from decisions.data import accepted, load_stream
from decisions.schema import labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["enron_spam", "phishing_email",
                        "customer_support", "ms_marco", "typed_decisions_all"])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--max-source-rows", type=int, default=128)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/additional_dataset_validation.json")
    args = parser.parse_args()
    if args.rows < 1 or args.max_source_rows < 1:
        parser.error("row bounds must be positive")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "protocol": "Bounded training-split schema checks; no training or evaluation scores",
              "streaming": True, "requested_rows": args.rows,
              "max_source_rows": args.max_source_rows, "datasets": {}}
    for name in args.datasets:
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / f"conf/dataset/{name}.yaml"), resolve=True)
        stream = load_stream(cfg, cfg["splits"]["train"])
        adapter = Adapter(cfg["schema"], stream.features)
        count, consumed, questions = 0, 0, 0
        types, targets, workflows = set(), Counter(), Counter()
        iterator = iter(stream)
        try:
            while count < args.rows and consumed < args.max_source_rows:
                original = next(iterator)
                consumed += 1
                if not accepted(original, cfg, "train"):
                    continue
                for row in adapter.iter_examples(original):
                    if not row["questions"]:
                        raise ValueError(f"{name}: no questions")
                    for qid, question in row["questions"].items():
                        probabilities = row["targets"][qid]
                        if set(probabilities) != set(labels(question)):
                            raise ValueError(f"{name}: mismatched target options")
                        values = list(probabilities.values())
                        if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-4):
                            raise ValueError(f"{name}: invalid target distribution")
                        types.add(question["type"])
                        targets[f"{qid}:{max(probabilities, key=probabilities.get)}"] += 1
                    questions += len(row["questions"])
                    if "workflow" in original:
                        workflows[original["workflow"]] += 1
                    count += 1
                    if count == args.rows:
                        break
        except StopIteration:
            pass
        finally:
            iterator.close()
        if count != args.rows:
            raise ValueError(f"{name}: only {count} accepted rows within the source bound")
        report["datasets"][name] = {"status": "ok", "rows": count, "source_rows": consumed,
            "questions": questions, "types": sorted(types), "target_counts": dict(targets),
            "workflows": dict(workflows), "source": cfg["source"], "splits": cfg["splits"],
            "holdout": cfg.get("holdout"), "validation_group_field": cfg.get("validation_group_field")}
        print(f"{name}: {count} decisions, {questions} questions from {consumed} streamed source rows", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
