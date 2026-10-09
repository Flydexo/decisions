"""Resume the verified public Decision Index 0.3 suite, one benchmark at a time on MPS."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

from decision_index import editions
from decision_index.suite.io import Suite


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/decision_index_03"
SUITE = BASE / "suite-0.3"
RUN = BASE / "runs/bad-laya-full-03"
ROWS = BASE / "per-benchmark-rows"
CHECKPOINT = ROOT / "outputs/rtx4090_curriculum/full_split_ordered/last_bf16_fresh_optimizer.pt"
OFFICIAL_MANIFEST = Path("/private/tmp/decision-index-0.3/hub/manifest.json")
LOCAL_MANIFEST = BASE / "work/artifacts/benchmark-suite/release-v1-rebuilt/manifest.json"
TEMPERATURE = 12.595144782442853
ENGINE = "decisions.benchmark:DecisionEngine"
MIN_FREE_BYTES = 4 * 1024**3


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def preflight() -> tuple[Suite, dict]:
    suite = Suite(SUITE, edition="0.3")
    verification = suite.verify(strict=True)
    if not verification["match"]:
        raise ValueError("The 0.3 suite does not match the official published hashes")
    local = {x["catalog_id"]: x for x in json.loads(LOCAL_MANIFEST.read_text())["benchmarks"]}
    official = {x["catalog_id"]: x for x in json.loads(OFFICIAL_MANIFEST.read_text())["benchmarks"]}
    mismatched = [n for n in official if local[n]["sources"] != official[n]["sources"]
                  or local[n]["selected_cases"] != official[n]["selected_cases"]]
    if mismatched:
        raise ValueError(f"Base source or selection mismatch: {mismatched}")
    return suite, {"suite_verification": verification, "base_manifest_mismatch": None}


def prepare_rows(suite: Suite) -> dict[int, dict]:
    ROWS.mkdir(parents=True, exist_ok=True)
    counts: Counter[int] = Counter()
    names: dict[int, str] = {}
    with ExitStack() as stack:
        streams = {}
        for row in suite.rows(apply_exclusions=True):
            e = row["_evaluation"]
            n = e["catalog_id"]
            if n not in streams:
                path = ROWS / f"{n:02d}.jsonl.gz"
                streams[n] = stack.enter_context(gzip.open(path.with_suffix(".gz.tmp"), "wt", encoding="utf-8", compresslevel=1))
            streams[n].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[n] += 1
            names[n] = e["dataset"]
    for n in counts:
        (ROWS / f"{n:02d}.jsonl.gz.tmp").replace(ROWS / f"{n:02d}.jsonl.gz")
    expected = editions.get("0.3")["scoreable"] + editions.get("0.3")["added_requests"]
    if sum(counts.values()) != expected:
        raise ValueError(f"Expected {expected} scoreable requests, found {sum(counts.values())}")
    return {n: {"catalog_id": n, "dataset": names[n], "requests": count,
                "rows": str(ROWS / f"{n:02d}.jsonl.gz")}
            for n, count in counts.items()}


def latest_results() -> dict[str, str]:
    path = RUN / "results.jsonl"
    if not path.exists():
        return {}
    records = {}
    with path.open() as stream:
        for line in stream:
            if line.endswith("\n"):
                row = json.loads(line)
                records[row["run_id"]] = row["status"]
    return records


def stage_status(path: Path, records: dict[str, str]) -> Counter[str]:
    statuses: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            statuses[records.get(json.loads(line)["_evaluation"]["run_id"], "pending")] += 1
    return statuses


def run_stage(stage: dict) -> None:
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("Less than 4 GiB free; stopping before the next benchmark")
    command = [sys.executable, "-m", "decision_index", "run", "--edition", "0.3",
               "--rows", stage["rows"], "--engine", ENGINE,
               "--option", f"checkpoint={CHECKPOINT}", "--option", "device=mps",
               "--option", "question_batch_size=8", "--option", f"temperature={TEMPERATURE}",
               "--compact", "--out", str(RUN)]
    with (RUN / f"{stage['catalog_id']:02d}.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    statuses = stage_status(Path(stage["rows"]), latest_results())
    if statuses["pending"] or statuses["error"] or sum(statuses.values()) != stage["requests"]:
        raise RuntimeError(f"Incomplete benchmark {stage['catalog_id']}: {dict(statuses)}")
    stage["statuses"] = dict(statuses)
    stage["finished_utc"] = stamp()


def score() -> dict:
    command = [sys.executable, "-m", "decision_index", "score", "--edition", "0.3",
               "--suite-dir", str(SUITE), "--results", str(RUN / "results.jsonl"),
               "--engine", ENGINE, "--out", str(RUN)]
    with (RUN / "score.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((RUN / "scores.json").read_text())
    return {"completed": result["completed"], "complete": result["complete"],
            "public_index": result["decision_index"] if result["complete"] else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "sequential.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite, verification = preflight()
        stages = prepare_rows(suite)
        if args.prepare_only:
            print(f"Prepared {len(stages)} benchmarks and {sum(s['requests'] for s in stages.values())} rows")
            return
        # Counted benchmarks first; within each group, finish smaller ones first.
        from decision_index.scoring import index02

        spec = index02.spec("0.3")
        uncounted = {entry["id"] for entry in spec["not_in_index"]}
        counted = set(stages) - uncounted
        order = sorted(stages, key=lambda n: (n not in counted, stages[n]["requests"], n))
        progress = {"started_utc": stamp(), "edition": "0.3", "engine": ENGINE,
                    "device": "mps", "checkpoint": str(CHECKPOINT),
                    "question_batch_size": 8, "temperature": TEMPERATURE,
                    "verification": verification, "order": order, "stages": stages}
        atomic_json(RUN / "sequential-progress.json", progress)
        for n in order:
            stage = stages[n]
            statuses = stage_status(Path(stage["rows"]), latest_results())
            if statuses["pending"] or statuses["error"]:
                progress["active"] = n
                atomic_json(RUN / "sequential-progress.json", progress)
                run_stage(stage)
            else:
                stage["statuses"] = dict(statuses)
                stage["finished_utc"] = stamp()
            progress.pop("active", None)
            progress["last_finished"] = n
            atomic_json(RUN / "sequential-progress.json", progress)
            print(f"Completed {n} {stage['dataset']}: {stage['statuses']}", flush=True)
        progress["score"] = score()
        progress["finished_utc"] = stamp()
        atomic_json(RUN / "sequential-progress.json", progress)
        print(json.dumps(progress["score"]), flush=True)


if __name__ == "__main__":
    main()
