"""Summarize saved 0.3 benchmark scores against a pinned public board snapshot.

This publishes aggregate metrics only. It does not copy benchmark requests or
turn a partial run into an overall public-index score.
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/decision_index_03"
SCORES = BASE / "runs/bad-laya-paused-03/scores.json"
PROGRESS = BASE / "runs/bad-laya-full-03/sequential-progress.json"
ARC = ROOT / "reports/bad-laya-arc-easy.json"
OUTPUT = ROOT / "reports/bad-laya-decision-index-03-paused.json"
BOARD_REPO = "multimodalart/jev-decision-index"
BOARD_REVISION = "e452ca53f88e735031ca0605559c7d83fd1aa1b6"
COMPARATORS = ("Laya", "Kev 0.8B r15", "Kev 4B r10", "Cloudflare clef")


def board_data() -> dict:
    try:
        path = hf_hub_download(BOARD_REPO, "data/index.json", repo_type="space",
                               revision=BOARD_REVISION, token=False, local_files_only=True)
    except LocalEntryNotFoundError:
        path = hf_hub_download(BOARD_REPO, "data/index.json", repo_type="space",
                               revision=BOARD_REVISION, token=False)
    return json.loads(Path(path).read_text())


def percentage(value: float | None) -> str:
    if value is None:
        return "—"
    amount = 100 * value
    return f"{amount:.2f}%" if 0 < amount < 1 else f"{amount:.1f}%"


def main() -> None:
    scores = json.loads(SCORES.read_text())
    progress = json.loads(PROGRESS.read_text())
    arc = json.loads(ARC.read_text())
    board = board_data()
    if scores["complete"]:
        raise ValueError("Expected the paused partial run, not a complete 0.3 result")
    models = {model["name"]: model for model in board["models"]}
    rows = []
    for catalog_id, item in scores["benchmarks"].items():
        if not (item["answered"] or item["unsupported"]):
            continue
        number = int(catalog_id)
        benchmark = board["benchmarks"][catalog_id]
        if item["requests"] != progress["stages"][catalog_id]["requests"]:
            raise ValueError(f"Case count differs for {item['dataset']}")
        if item["errors"] or item["abstained"]:
            raise ValueError(f"Unexpected errors or abstentions for {item['dataset']}")
        if item["metric"] and item["metric"] != benchmark["metric"]:
            raise ValueError(f"Metric differs for {item['dataset']}")
        if abs(item["chance"] - benchmark["baseline"]["value"]) > 0.0001:
            raise ValueError(f"Chance reference differs for {item['dataset']}")
        rows.append({
            "catalog_id": number, "dataset": item["dataset"],
            "metric": item["metric"] or benchmark["metric"],
            "requests": item["requests"], "answered": item["answered"],
            "unsupported": item["unsupported"], "pending": item["pending"],
            "status": "partial" if item["pending"] else "complete",
            "score_on_answered": item["score"],
            "coverage_adjusted_score": item["index_raw"] if item["answered"] else None,
            "chance": item["chance"], "in_index": item["in_index"],
        })
    rows.append({
        "catalog_id": 26, "dataset": "ARC-Easy", "metric": "accuracy",
        "requests": arc["cases"], "answered": arc["answered"],
        "unsupported": 0, "pending": 0, "status": "complete, separate run",
        "score_on_answered": arc["accuracy"],
        "coverage_adjusted_score": arc["accuracy"],
        "chance": arc["chance"], "in_index": False,
    })
    rows.sort(key=lambda row: row["catalog_id"])
    for row in rows:
        key = str(row["catalog_id"])
        peers = {}
        for name in COMPARATORS:
            model = models[name]
            result = model["results"].get(key)
            if not result or result["requests"] != row["requests"]:
                peers[name] = None
                continue
            if result["metric"] != row["metric"]:
                raise ValueError(f"{name} metric differs for {row['dataset']}")
            published = model["benchmarks"].get(key)
            peers[name] = (published["raw"] if published else
                           result["score"] * result["answered"] / result["requests"])
        row["comparators"] = peers
    report = {
        "model": "flydexo/bad-laya", "edition": "0.3", "complete": False,
        "canonical_suite_verified": True, "completed_requests": scores["completed"],
        "complete_benchmarks_in_sequential_run": sum(
            bool(stage.get("finished_utc")) for stage in progress["stages"].values()),
        "board_snapshot": {"repo": BOARD_REPO, "revision": BOARD_REVISION,
                           "generated_utc": board["generated_utc"]},
        "comparators": list(COMPARATORS), "benchmarks": rows,
        "note": "Completed rows use the kit's coverage-adjusted index_raw metric. ACOS is partial and is not comparable with full peer scores. A complete public index is unavailable.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("| Benchmark / metric | bad-laya | Answered | Chance / baseline | Laya | Kev 0.8B | Kev 4B | Cloudflare clef |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for row in rows:
        name = row["dataset"]
        if row["catalog_id"] == 64:
            name = "New Yorker captions"
        if row["catalog_id"] == 9:
            name = "Home appliance"
        if not row["in_index"]:
            name += " (shown only)"
        if row["status"] == "partial":
            name += " (partial)"
        cells = [f"{name} · {row['metric']}",
                 percentage(row["coverage_adjusted_score"]),
                 f"{row['answered']}/{row['requests']}", percentage(row["chance"])]
        cells += [percentage(row["comparators"][peer]) for peer in COMPARATORS]
        print("| " + " | ".join(cells) + " |")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
