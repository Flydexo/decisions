"""Compare locally rebuilt 0.3 requests with a complete public 0.2.1 run.

Edition 0.3 reuses 0.2.1 base requests except for GSM8K. This audit compares
run IDs and payload hashes without copying benchmark inputs into the report.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from decision_index import editions
from decision_index.scoring import index02
from decision_index.suite.io import Suite
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "outputs/decision_index_03/suite-0.3-provisional"
OUTPUT = ROOT / "reports/decision-index-03-suite-audit.json"
REPO = "Lukitaduarte/dinah-0-decision-index-results"
REVISION = "f1c4c6f50d7ff6721022aa0d531f2d6f36346f18"
PREFIX = "runs/dinah-0/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    scores_path = Path(hf_hub_download(REPO, PREFIX + "scores.json", repo_type="dataset",
                                       revision=REVISION))
    results_path = Path(hf_hub_download(REPO, PREFIX + "results.jsonl", repo_type="dataset",
                                        revision=REVISION))
    reference_scores = json.loads(scores_path.read_text())
    if (reference_scores["edition"] != "0.2.1" or not reference_scores["complete"]
            or reference_scores["suite"]["rows_sha256"] != editions.get("0.3")["rows_sha256"]):
        raise ValueError("Reference run is not complete on the canonical shared base rows")
    reference = {}
    with results_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            reference[row["run_id"]] = row["payload_sha256"]

    suite = Suite(SUITE, edition="0.3")
    verification = suite.verify(strict=False)
    if not all(verification[key] for key in
               ("added_match", "gsm8k_match", "exclusions_match", "subsets_match")):
        raise ValueError("Added, GSM8K, or exclusion hashes do not match the kit")
    counts = defaultdict(lambda: {"requests": 0, "matching_ids": 0, "matching_payload_sha256": 0})
    for row in suite.rows(apply_exclusions=True):
        entry = row["_evaluation"]
        number = entry["catalog_id"]
        if number >= 56 or number == 30:
            continue  # New 0.2 benchmarks and rebuilt 0.3 GSM8K are hash-checked separately.
        item = counts[number]
        item["requests"] += 1
        if entry["run_id"] in reference:
            item["matching_ids"] += 1
            item["matching_payload_sha256"] += (
                reference[entry["run_id"]] == entry["payload_sha256"])
    mismatches = [number for number, item in counts.items()
                  if item["matching_payload_sha256"] != item["requests"]]
    if mismatches != [6] or counts[6] != {"requests": 10000, "matching_ids": 10000,
                                         "matching_payload_sha256": 0}:
        raise ValueError(f"Unexpected shared-base mismatch: {mismatches}")
    uncounted = {entry["id"] for entry in index02.spec("0.3")["not_in_index"]}
    if 6 not in uncounted:
        raise ValueError("RouterBench must be excluded from the 0.3 public index")
    totals = {key: sum(item[key] for item in counts.values())
              for key in ("requests", "matching_ids", "matching_payload_sha256")}
    report = {
        "edition": "0.3",
        "reference": {"dataset": REPO, "revision": REVISION,
                      "results_path": PREFIX + "results.jsonl",
                      "results_sha256": sha256(results_path),
                      "scores_path": PREFIX + "scores.json",
                      "scores_sha256": sha256(scores_path),
                      "edition": "0.2.1", "complete": True,
                      "base_rows_sha256": reference_scores["suite"]["rows_sha256"]},
        "comparison_scope": "Shared 0.2.1 base requests still used in 0.3; excludes new benchmarks and replaced GSM8K rows",
        "shared_base_totals": totals,
        "by_catalog_id": {str(number): item for number, item in sorted(counts.items())},
        "routerbench": {"catalog_id": 6, "requests": counts[6]["requests"],
                        "matching_ids": counts[6]["matching_ids"],
                        "matching_payload_sha256": counts[6]["matching_payload_sha256"],
                        "in_public_index": False},
        "all_other_shared_base_payloads_match": True,
        "added_rows_sha256_match": verification["added_match"],
        "gsm8k_rows_sha256_match": verification["gsm8k_match"],
        "full_base_file_sha256_match": verification["match"],
        "canonical_suite_verified": False,
        "official_submission_ready": False,
        "note": "All shared base request IDs match. Only RouterBench payload hashes differ; it is shown but not counted. The base file still fails the official frozen hash, so a completed local score remains provisional.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Matched {totals['matching_payload_sha256']}/{totals['requests']} shared base payloads; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
