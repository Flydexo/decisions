"""Export the evaluated BF16 curriculum checkpoint for flydexo/bad-laya.

The Hub artifact contains inference weights and tokenizer files, not the
checkpoint's optimizer, RNG state, dataset configuration, or local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/rtx4090_curriculum/full_split_ordered/last_bf16_fresh_optimizer.pt"
DEFAULT_OUTPUT = ROOT / "outputs/huggingface/bad-laya"
CARD = ROOT / "hf/bad-laya"
CALIBRATION = ROOT / "reports/bad-laya-calibration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(checkpoint)
    weights_path, config_path = output / "model.safetensors", output / "decision_config.json"
    if config_path.exists() and weights_path.exists():
        old = json.loads(config_path.read_text())
        reusable = old.get("source_checkpoint_sha256") == source_hash and old.get("weights_sha256") == sha256(weights_path)
    else:
        reusable = False
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if saved.get("format_version") != 2 or not saved.get("encoder") or not saved.get("head"):
        raise ValueError("Expected a full-encoder format-2 decision checkpoint")
    if not reusable:
        tensors = {f"bert.{key}": value.detach().contiguous()
                   for key, value in saved["encoder"].items()}
        tensors.update({key: value.detach().contiguous() for key, value in saved["head"].items()})
        temporary = weights_path.with_suffix(".safetensors.tmp")
        try:
            save_file(tensors, str(temporary), metadata={"format": "pt"})
            temporary.replace(weights_path)
        finally:
            temporary.unlink(missing_ok=True)
    progress = saved["progress"]
    model = saved["config"]["model"]
    config = {
        "format_version": 1,
        "base_model": model["encoder"],
        "base_revision": model["revision"],
        "model": model,
        "ablation": saved["config"]["ablation"],
        "checkpoint_step": progress["step"],
        "completed_stages": progress["stage_index"],
        "partial_stage_rows": progress["rows_in_stage"],
        "source_checkpoint_sha256": source_hash,
        "weights_sha256": sha256(weights_path),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if CALIBRATION.exists():
        report = json.loads(CALIBRATION.read_text())
        if report["checkpoint_sha256"] != source_hash:
            raise ValueError("Calibration belongs to a different checkpoint")
        (output / "calibration.json").write_text(json.dumps({
            "method": "single temperature minimizing pooled validation negative log-likelihood",
            "temperature": report["temperature"],
            "checkpoint_sha256": source_hash,
            "fit_questions": report["fit"]["raw"]["questions"],
            "held_out_eval_questions": report["test"]["raw"]["questions"],
            "held_out_eval_raw": {key: report["test"]["raw"][key] for key in
                                  ("accuracy", "mean_chosen_probability", "nll", "brier", "probability_ece")},
            "held_out_eval_calibrated": {key: report["test"]["calibrated"][key] for key in
                                         ("accuracy", "mean_chosen_probability", "nll", "brier", "probability_ece")},
        }, indent=2, allow_nan=False) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(model["encoder"], revision=model["revision"])
    tokenizer.save_pretrained(output / "tokenizer")
    shutil.copy2(CARD / "README.md", output / "README.md")
    (output / "assets").mkdir(exist_ok=True)
    shutil.copy2(CARD / "assets/overview.png", output / "assets/overview.png")
    print(json.dumps({"output": str(output), "weights_bytes": weights_path.stat().st_size,
                      "weights_sha256": config["weights_sha256"],
                      "source_checkpoint_sha256": source_hash}, indent=2))


if __name__ == "__main__":
    main()
