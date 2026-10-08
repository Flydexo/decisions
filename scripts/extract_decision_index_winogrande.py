"""Select the 1,267 WinoGrande requests from an imported Decision Index 0.3 suite."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from decision_index.suite.io import Suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [r for r in Suite(args.suite_dir, edition="0.3").rows(apply_exclusions=True)
            if r["_evaluation"]["catalog_id"] == 28]
    ids = [r["_evaluation"]["run_id"] for r in rows]
    if len(rows) != 1267 or len(set(ids)) != 1267:
        raise ValueError(f"Expected 1,267 distinct WinoGrande requests, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
