"""Train all configured datasets by size, evaluating and checkpointing after each."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from decisions.cuda_pilot import check_environment
from decisions.curriculum import SOURCE_TRAIN_ROWS, ordered_sources, train_curriculum
from decisions.data import dataset_configs
from scripts.run_rtx4090_pilot import worker_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "outputs/rtx4090_curriculum/full_split_ordered")
    parser.add_argument("--hours", type=float, default=8,
                        help="Total training wall-clock budget across resumes; increase to continue")
    parser.add_argument("--plan", action="store_true", help="Show the full-split stage order without GPU or data access")
    parser.add_argument("--check-env", action="store_true", help="Check the RTX 4090 environment without training")
    parser.add_argument("--resume", action="store_true", help="Resume the exact last.pt optimizer checkpoint")
    args = parser.parse_args()
    if not 0 < args.hours < 1e6:
        parser.error("--hours must be positive")
    directory = args.run_dir.resolve()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=rtx4090_curriculum"])
    cfg.training.max_seconds = int(args.hours * 3600)
    if args.resume:
        cfg.resume = str(directory / "last.pt")
    config = OmegaConf.to_container(cfg, resolve=True)
    datasets = dataset_configs(config)
    names = [dataset["name"] for dataset in datasets]
    order = ordered_sources(names)
    if len(names) != len(order) or any(name in {"ms_marco", "commitpackft"} for name in names):
        parser.error("Sources must be unique and exclude MS MARCO and CommitPackFT")
    if config["training"].get("validation_role") != "validation" or not config["data"]["separate_validation"]:
        parser.error("Stage evaluation requires separate validation partitions")
    if args.plan:
        print(json.dumps({"run_dir": str(directory), "source_count": len(order),
            "published_train_rows": sum(SOURCE_TRAIN_ROWS[name] for name in order),
            "stages": [{"stage": index + 1, "dataset": name,
                        "published_train_rows": SOURCE_TRAIN_ROWS[name]}
                       for index, name in enumerate(order)],
            "evaluation_datasets_after_each_stage": order,
            "config": config}, indent=2))
        return
    if args.resume and not (directory / "last.pt").is_file():
        parser.error("Resume requires last.pt in the run directory")
    if (directory / "last.pt").exists() and not args.resume and not args.check_env:
        parser.error("Existing training checkpoint: use --resume or a new --run-dir")
    directory.mkdir(parents=True, exist_ok=True)
    with worker_lock(directory):
        environment = check_environment(config, directory)
        (directory / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        if args.check_env:
            print(json.dumps(environment, indent=2))
            return
        train_curriculum(cfg, directory)


if __name__ == "__main__":
    main()
