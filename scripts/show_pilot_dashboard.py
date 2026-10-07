"""Open the saved RLCD pilot dashboard using its explicit local database."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=root / "outputs/collapse_rlcd_pilot")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify saved runs without starting a server")
    args = parser.parse_args()
    directory = args.run_dir.resolve() / "trackio"
    project = "decisions-collapse-rlcd"
    if not (directory / f"{project}.db").is_file():
        parser.error(f"Pilot database not found in {directory}")

    # Trackio reads this variable at import time, before the dashboard starts.
    os.environ["TRACKIO_DIR"] = str(directory)
    import trackio
    from trackio.sqlite_storage import SQLiteStorage

    runs = SQLiteStorage.get_runs(project)
    print(f"Database directory: {directory}", flush=True)
    print(f"Project: {project}; runs: {', '.join(runs)}", flush=True)
    if not args.check:
        trackio.show(project=project, host="127.0.0.1", server_port=args.port,
                     share=False, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
