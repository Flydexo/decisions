"""Summarize the single WinoGrande Decision Index 0.3 run for publication.

The report contains aggregate scores only; benchmark questions are not published.
"""
from __future__ import annotations

import gzip
import json
from math import sqrt
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/decision_index_03/runs/bad-laya-winogrande"
ROWS = ROOT / "outputs/decision_index_03/winogrande-0.3.jsonl.gz"
LOCAL_MANIFEST = ROOT / "outputs/decision_index_03/work/artifacts/benchmark-suite/release-v1-rebuilt/manifest.json"
OFFICIAL_MANIFEST = Path("/private/tmp/decision-index-0.3/hub/manifest.json")
OUTPUT = ROOT / "reports/bad-laya-winogrande.json"
COMPARATORS = (
    "Cloudflare clef", "Kev 27B", "Kev 9B v2", "Kev 4B r10",
    "Kev 0.8B r15", "Laya",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def benchmark(manifest: dict) -> dict:
    return next(entry for entry in manifest["benchmarks"] if entry["catalog_id"] == 28)


def main() -> None:
    scores = read_json(RUN / "scores.json")
    winogrande = scores["benchmarks"]["28"]
    if scores["complete"]:
        raise ValueError("Expected a single-benchmark run, not a complete suite")

    local = benchmark(read_json(LOCAL_MANIFEST))
    official = benchmark(read_json(OFFICIAL_MANIFEST))
    if (local["sources"] != official["sources"] or
            local["selected_cases"] != official["selected_cases"] or
            local["selected_cases"] != local["available_cases"]):
        raise ValueError("Local WinoGrande source or case count differs from the official manifest")

    with gzip.open(ROWS, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    results = {r["run_id"]: r for r in map(json.loads, (RUN / "results.jsonl").read_text().splitlines())}
    if len(rows) != len(results) or len(rows) != 1267:
        raise ValueError("WinoGrande run must contain exactly 1,267 distinct questions and results")
    correct = 0
    for row in rows:
        record = results[row["_evaluation"]["run_id"]]
        if record["status"] != "ok" or record["payload_sha256"] != row["_evaluation"]["payload_sha256"]:
            raise ValueError("Missing, unsuccessful, or mismatched WinoGrande result")
        correct += record["response"]["answers"]["q1"]["choice"] == row["expected"]["q1"]
    accuracy = correct / len(rows)
    if abs(accuracy - winogrande["score"]) > 0.0001 or winogrande["answered"] != len(rows):
        raise ValueError("Direct accuracy and official kit score disagree")

    source = Path(hf_hub_download(
        "multimodalart/jev-decision-index", "data/index.json", repo_type="space", token=False,
    ))
    published = read_json(source)
    board = published["benchmarks"]["28"]
    if board["cases"] != len(rows) or board["metric"] != "accuracy":
        raise ValueError("Published board uses a different WinoGrande case count or metric")
    models = {model["name"]: model for model in published["models"]}
    comparators = []
    for name in COMPARATORS:
        model = models[name]
        result = model["results"]["28"]
        if result["requests"] != len(rows) or result["answered"] != len(rows):
            raise ValueError(f"Comparator {name} has incomplete WinoGrande coverage")
        comparators.append({"name": name, "accuracy": model["benchmarks"]["28"]["raw"]})

    # Wilson 95% interval for the one observed accuracy, with no paired claim.
    n, z = len(rows), 1.96
    midpoint = (accuracy + z*z/(2*n)) / (1+z*z/n)
    radius = z*sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = {
        "model": "flydexo/bad-laya", "edition": "0.3", "dataset": "WinoGrande",
        "catalog_id": 28, "cases": n, "answered": n, "coverage": 1.0,
        "correct": correct, "accuracy": accuracy, "accuracy_wilson_95": [midpoint-radius, midpoint+radius],
        "chance": board["baseline"]["value"],
        "jev_accuracy": published["jev"]["benchmarks"]["28"]["raw"],
        "comparators": comparators,
        "comparator_snapshot": {
            "url": "https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json",
            "revision": source.parents[1].name,
            "generated_utc": published["generated_utc"],
        },
        "kit_revision": "9eb2dbe",
        "winogrande_source_sha256": local["sources"][0]["sha256"],
        "winogrande_source_matches_official": True,
        "full_suite_verified": False,
        "full_suite_score": None,
        "calibration_temperature": read_json(ROOT / "reports/bad-laya-calibration.json")["temperature"],
        "checkpoint_sha256": read_json(ROOT / "reports/curriculum-results-data.json")["checkpoint_sha256"],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{correct}/{n} = {accuracy:.4%}; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
