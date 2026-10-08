"""Build the standalone RTX 4090 curriculum results page.

Use --refresh to snapshot the local checkpoint evaluation and validation
samples. Without it, rebuild the HTML from the tracked data snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
RUN = ROOT / "outputs/rtx4090_curriculum/full_split_ordered"
SNAPSHOT = REPORTS / "curriculum-results-data.json"
PAGE = REPORTS / "curriculum-results.html"
TEMPLATE = REPORTS / "curriculum-results.template.html"

NAMES = {
    "ag_news": "AG News", "boolq": "BoolQ", "sst5": "SST-5",
    "arc_challenge": "ARC-Challenge", "banking77": "Banking77",
    "mnli": "MultiNLI", "yelp_review_full": "Yelp Review Full", "trec": "TREC",
    "dbpedia14": "DBpedia 14", "amazon_reviews_multi_en": "Amazon Reviews · EN",
    "imdb": "IMDb", "openbookqa": "OpenBookQA", "commonsenseqa": "CommonsenseQA",
    "aegis": "Aegis Safety 2.0", "consumer_finance": "Consumer Finance",
    "codereviewer": "CodeReviewer", "flakeflagger": "FlakeFlagger",
    "enron_spam": "Enron Spam", "phishing_email": "Phishing Email",
    "customer_support": "Customer Support Tickets",
    "typed_decisions_all": "Typed Decisions · All workflows",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def random_tail(probabilities: list[float], correct: int) -> float:
    """Exact P(random guesses score at least this many correct)."""
    distribution = [1.0]
    for probability in probabilities:
        next_distribution = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            next_distribution[count] += mass * (1 - probability)
            next_distribution[count + 1] += mass * probability
        distribution = next_distribution
    return sum(distribution[correct:])


def refresh_snapshot() -> dict:
    import torch
    from transformers import AutoTokenizer

    from decisions.evaluation import preprocessing
    from decisions.schema import Unsupported, prepare_request

    latest = read_json(RUN / "local_evaluation/latest_bf16.json")
    if latest["completed_sources"] != 21 or latest["errors"]:
        raise ValueError("The local evaluation must contain all 21 sources without errors")
    pilot = read_json(REPORTS / "rtx4090_pilot_baseline_summary.json")
    checkpoint = RUN / "last_bf16_fresh_optimizer.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = saved["config"]
    progress = saved["progress"]
    order = config["training"]["curriculum_order"]
    previous_name = order[progress["stage_index"] - 1]
    previous = read_json(RUN / "stages" / f"{progress['stage_index']:02d}-{previous_name}.json")
    expected = set(order)
    if not (set(latest["validation"]) == set(pilot["selected_validation"]["datasets"])
            == set(previous["validation"]) == expected):
        raise ValueError("The latest, pilot, and previous reports must cover identical datasets")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if latest["identity"]["checkpoint_sha256"] != digest.hexdigest():
        raise ValueError("Local evaluation belongs to a different checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"], revision=config["model"].get("revision"))
    settings = preprocessing(config["model"], config["evaluation"]["strict"])
    datasets = []
    for index, name in enumerate(order, start=1):
        cache_files = glob.glob(str(RUN / "local_evaluation/validation_samples" / f"{name}-*.json"))
        if len(cache_files) != 1:
            raise ValueError(f"Expected one validation sample cache for {name}: {cache_files}")
        samples = read_json(Path(cache_files[0]))["rows"]
        probabilities = []
        for sample in samples:
            try:
                inputs = prepare_request(sample, tokenizer, **settings)
            except Unsupported:
                continue
            probabilities.extend(1 / len(labels) for labels in inputs["option_labels"])
        measured = latest["validation"][name]
        if len(probabilities) != measured["questions"]:
            raise ValueError(f"Random reference and model question counts disagree for {name}")
        correct = round(measured["accuracy"] * len(probabilities))
        datasets.append({
            "id": name, "name": NAMES[name], "stage": index,
            "training_status": ("completed" if index <= progress["stage_index"] else
                                "partial" if index == progress["stage_index"] + 1 else "unseen"),
            "questions": measured["questions"], "rows": measured["rows"],
            "latest": measured["accuracy"], "latest_nll": measured["nll"],
            "chance": sum(probabilities) / len(probabilities),
            "random_tail_probability": random_tail(probabilities, correct),
            "pilot": pilot["selected_validation"]["datasets"][name]["accuracy"],
            "previous": previous["validation"][name]["accuracy"],
            "previous_nll": previous["validation"][name]["nll"],
        })
    history = []
    for index, name in enumerate(order[:progress["stage_index"]], start=1):
        stage = read_json(RUN / "stages" / f"{index:02d}-{name}.json")
        history.append({"stage": index, "name": NAMES[name],
                        "accuracy": stage["macro_validation_accuracy"], "partial": False})
    history.append({"stage": progress["stage_index"] + 1,
                    "name": NAMES[order[progress["stage_index"]]],
                    "accuracy": latest["macro_validation_accuracy"], "partial": True})
    macro = lambda field: sum(row[field] for row in datasets) / len(datasets)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": latest["identity"]["checkpoint_sha256"],
        "checkpoint_step": progress["step"], "completed_stages": progress["stage_index"],
        "total_stages": len(order), "current_stage": NAMES[order[progress["stage_index"]]],
        "current_stage_rows": progress["rows_in_stage"],
        "source_count": len(datasets), "question_count": sum(row["questions"] for row in datasets),
        "macro": {"latest": macro("latest"), "chance": macro("chance"),
                  "pilot": macro("pilot"), "previous": macro("previous")},
        "mean_nll": {"latest": macro("latest_nll"), "previous": macro("previous_nll")},
        "counts": {"above_chance": sum(row["latest"] > row["chance"] for row in datasets),
                   "chance_p_below_005": sum(row["random_tail_probability"] < .05 for row in datasets),
                   "above_pilot": sum(row["latest"] > row["pilot"] for row in datasets),
                   "tied_pilot": sum(row["latest"] == row["pilot"] for row in datasets),
                   "above_previous": sum(row["latest"] > row["previous"] for row in datasets)},
        "datasets": datasets, "history": history,
        "sources": ["outputs/rtx4090_curriculum/full_split_ordered/local_evaluation/latest_bf16.json",
                    "reports/rtx4090_pilot_baseline_summary.json",
                    f"outputs/rtx4090_curriculum/full_split_ordered/stages/{progress['stage_index']:02d}-{previous_name}.json",
                    "outputs/rtx4090_curriculum/full_split_ordered/local_evaluation/validation_samples/"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Recompute snapshot from local validation files")
    args = parser.parse_args()
    if args.refresh:
        data = refresh_snapshot()
        SNAPSHOT.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        data = read_json(SNAPSHOT)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    template = TEMPLATE.read_text()
    if template.count("__CURRICULUM_DATA__") != 1:
        raise ValueError("Template must have one data placeholder")
    PAGE.write_text(template.replace("__CURRICULUM_DATA__", payload))
    print(f"Wrote {SNAPSHOT} and {PAGE}" if args.refresh else f"Wrote {PAGE}")


if __name__ == "__main__":
    main()
