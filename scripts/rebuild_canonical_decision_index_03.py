"""Rebuild the canonical 0.3 suite with RouterBench's released fsum averages.

The pinned kit computes RouterBench calibration means with plain ``sum``;
the released normalized hashes were produced with ``math.fsum``. All other
builder code and source files remain the pinned kit's. No benchmark rows are
committed or redistributed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from decision_index.suite.build import adapters_scored, rebuild
from decision_index.suite.build.layout import Layout
from decision_index.suite.download import from_local
from decision_index.suite.io import Suite


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/decision_index_03"
WORK = BASE / "work"
SUITE = BASE / "suite-0.3"
OFFICIAL_MANIFEST = Path("/private/tmp/decision-index-0.3/hub/manifest.json")
EDITION_MANIFEST = Path("/private/tmp/decision-index-0.3/hub/0.3/manifest.json")
EXCLUSIONS = Path("/private/tmp/decision-index-0.3/hub/excluded-questions.json")
CANONICAL_ROUTER_ROWS = BASE / "per-benchmark-rows/06.jsonl.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebuild_router() -> None:
    official = json.loads(OFFICIAL_MANIFEST.read_text())
    entry = next(x for x in official["benchmarks"] if x["catalog_id"] == 6)
    expected = {Path(s["path"]).name: s["sha256"] for s in entry["sources"]}
    layout = Layout(WORK)
    if not (layout.raw / "routerbench/routerbench_0shot.pkl").exists():
        raise FileNotFoundError("Pinned RouterBench raw files are missing")
    with tempfile.TemporaryDirectory(prefix="decision-index-router-fsum-") as tmp:
        temp = Layout(tmp)
        (temp.raw / "routerbench").symlink_to(layout.raw / "routerbench", target_is_directory=True)
        old = getattr(adapters_scored, "sum", None)
        adapters_scored.sum = math.fsum
        try:
            adapters_scored.routerbench(temp)
        finally:
            if old is None:
                delattr(adapters_scored, "sum")
            else:
                adapters_scored.sum = old
        for name, digest in expected.items():
            source = temp.normalized / name
            if sha256(source) != digest:
                raise ValueError(f"RouterBench {name} still differs from the official hash")
            target = layout.normalized / name
            staged = target.with_suffix(target.suffix + ".verified.tmp")
            shutil.copyfile(source, staged)
            staged.replace(target)


def main() -> None:
    rebuild_router()
    with (BASE / "canonical-rebuild.log").open("w") as log:
        result = rebuild.main(WORK, skip_download=True, skip_normalize=True, edition="0.3",
                              exclusions=EXCLUSIONS,
                              log=lambda event: print(event, file=log, flush=True))
    if not (result["release_v1"]["byte_identical"] and result["rows_byte_identical"]
            and result["added_byte_identical"] and result["gsm8k"]["byte_identical"]):
        raise ValueError("Canonical 0.3 suite did not match every pinned row hash")
    out = Path(result["out"])
    imported = from_local(SUITE, out / "selected-rows.jsonl.gz",
                          added=out / "added-rows.jsonl.gz",
                          gsm8k=WORK / "artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz",
                          exclusions=EXCLUSIONS, manifest=EDITION_MANIFEST,
                          verify=True, edition="0.3")
    if not imported["match"]:
        raise ValueError("Canonical suite import did not verify")
    canonical = Suite(SUITE, edition="0.3")
    CANONICAL_ROUTER_ROWS.parent.mkdir(parents=True, exist_ok=True)
    staged = CANONICAL_ROUTER_ROWS.with_suffix(".gz.verified.tmp")
    count = 0
    with gzip.open(staged, "wt", encoding="utf-8", compresslevel=1) as stream:
        for row in canonical.rows(apply_exclusions=True):
            if row["_evaluation"]["catalog_id"] == 6:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    if count != 10000:
        raise ValueError(f"Expected 10,000 canonical RouterBench requests, found {count}")
    staged.replace(CANONICAL_ROUTER_ROWS)
    report = {"edition": "0.3", "kit_revision": "9eb2dbe",
              "routerbench_average": "math.fsum", "routerbench_normalized_sha256": {
                  name: sha256(Layout(WORK).normalized / name)
                  for name in ("RouterBench-0shot.jsonl", "RouterBench-5shot.jsonl")},
              "release_v1_byte_identical": True, "release_v2_byte_identical": True,
              "added_byte_identical": True, "gsm8k_byte_identical": True,
              "suite_verification": imported,
              "routerbench_rows_replaced_for_sequential_run": count,
              "note": "The active sequential runner began on the provisional suite; all prior non-RouterBench requests have identical canonical payload hashes. Score the final results against suite-0.3."}
    (BASE / "canonical-suite-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    public_report = {
        "edition": report["edition"],
        "kit_revision": report["kit_revision"],
        "routerbench_average": report["routerbench_average"],
        "routerbench_normalized_sha256": report["routerbench_normalized_sha256"],
        "release_v1_byte_identical": report["release_v1_byte_identical"],
        "release_v2_byte_identical": report["release_v2_byte_identical"],
        "added_byte_identical": report["added_byte_identical"],
        "gsm8k_byte_identical": report["gsm8k_byte_identical"],
        "suite": {key: imported[key] for key in (
            "expected_sha256", "match", "uncompressed_sha256",
            "uncompressed_match", "added_sha256", "added_match",
            "gsm8k_sha256", "gsm8k_match", "exclusions_sha256",
            "exclusions_match", "subsets_match")},
        "scope": "Public Decision Index 0.3 suite; private board tests are not included.",
    }
    (ROOT / "reports/decision-index-03-canonical-suite.json").write_text(
        json.dumps(public_report, indent=2) + "\n")
    print("Verified canonical 0.3 suite and replaced 10,000 future RouterBench request rows")


if __name__ == "__main__":
    main()
