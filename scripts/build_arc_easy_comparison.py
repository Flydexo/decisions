"""Publish aggregate ARC-Easy Decision Index 0.3 results and board comparators."""
from __future__ import annotations

import gzip
import json
from math import sqrt
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/decision_index_03/runs/bad-laya-arc-easy"
ROWS = ROOT / "outputs/decision_index_03/arc-easy-0.3.jsonl.gz"
LOCAL_MANIFEST = ROOT / "outputs/decision_index_03/work/artifacts/benchmark-suite/release-v1-rebuilt/manifest.json"
OFFICIAL_MANIFEST = Path("/private/tmp/decision-index-0.3/hub/manifest.json")
OUTPUT = ROOT / "reports/bad-laya-arc-easy.json"
COMPARATORS = (
    ("Cloudflare clef", "Cloudflare clef"),
    ("Kev 4B r10", "Kev 4B r10"),
    ("Kev 0.8B r15", "Kev 0.8B r15"),
    ("LiquidAI d1-omni-600M", "LiquidAI 600M"),
    ("Bekko System One v0 68M", "Bekko 68M"),
    ("Bekko System One v0 17M", "Bekko 17M"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def benchmark(manifest: dict) -> dict:
    return next(entry for entry in manifest["benchmarks"] if entry["catalog_id"] == 26)


def main() -> None:
    scores = read_json(RUN / "scores.json")
    arc = scores["benchmarks"]["26"]
    if scores["complete"]:
        raise ValueError("Expected a single-benchmark run, not a complete suite")

    local = benchmark(read_json(LOCAL_MANIFEST))
    official = benchmark(read_json(OFFICIAL_MANIFEST))
    if (local["sources"] != official["sources"] or
            local["selected_cases"] != official["selected_cases"] or
            local["selected_cases"] != local["available_cases"]):
        raise ValueError("Local ARC-Easy source or case count differs from the official manifest")

    with gzip.open(ROWS, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    results = {r["run_id"]: r for r in map(json.loads, (RUN / "results.jsonl").read_text().splitlines())}
    if len(rows) != len(results) or len(rows) != 2376:
        raise ValueError("ARC-Easy run must contain exactly 2,376 distinct requests and results")
    correct = 0
    for row in rows:
        record = results[row["_evaluation"]["run_id"]]
        if record["status"] != "ok" or record["payload_sha256"] != row["_evaluation"]["payload_sha256"]:
            raise ValueError("Missing, unsuccessful, or mismatched ARC-Easy result")
        correct += record["response"]["answers"]["q1"]["choice"] == row["expected"]["q1"]
    accuracy = correct / len(rows)
    if abs(accuracy - arc["score"]) > 0.0001 or arc["answered"] != len(rows):
        raise ValueError("Direct accuracy and official kit score disagree")

    source = Path(hf_hub_download(
        "multimodalart/jev-decision-index", "data/index.json", repo_type="space", token=False,
    ))
    published = read_json(source)
    board = published["benchmarks"]["26"]
    if board["cases"] != len(rows) or board["metric"] != "accuracy":
        raise ValueError("Published board uses a different ARC-Easy case count or metric")
    models = {model["name"]: model for model in published["models"]}
    comparators = []
    for name, display_name in COMPARATORS:
        model = models[name]
        result = model["results"]["26"]
        if result["requests"] != len(rows) or result["answered"] != len(rows):
            raise ValueError(f"Comparator {name} has incomplete ARC-Easy coverage")
        comparators.append({"name": name, "display_name": display_name,
                            "accuracy": model["benchmarks"]["26"]["raw"]})

    n, z = len(rows), 1.96
    midpoint = (accuracy + z*z/(2*n)) / (1+z*z/n)
    radius = z*sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = {
        "model": "flydexo/bad-laya", "edition": "0.3", "dataset": "ARC-Easy",
        "catalog_id": 26, "cases": n, "answered": n, "coverage": 1.0,
        "correct": correct, "accuracy": accuracy, "accuracy_wilson_95": [midpoint-radius, midpoint+radius],
        "chance": board["baseline"]["value"],
        "in_index": arc["in_index"],
        "jev_accuracy": published["jev"]["benchmarks"].get("26", {}).get("raw"),
        "comparators": comparators,
        "comparator_snapshot": {
            "url": "https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json",
            "revision": source.parents[1].name,
            "generated_utc": published["generated_utc"],
        },
        "kit_revision": "9eb2dbe",
        "arc_easy_source_sha256": local["sources"][0]["sha256"],
        "arc_easy_source_matches_official": True,
        "full_suite_verified": False, "full_suite_score": None,
        "selection_note": "Chosen before this run based on earlier ARC-Challenge validation and a short single-question request format.",
        "device": "mps",
        "calibration_temperature": read_json(ROOT / "reports/bad-laya-calibration.json")["temperature"],
        "checkpoint_sha256": read_json(ROOT / "reports/curriculum-results-data.json")["checkpoint_sha256"],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{correct}/{n} = {accuracy:.4%}; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
