"""Summarize recorded synthetic memory checks; never start training."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    checks = []
    for path in REPORTS.glob("finetune_preflight_*.json"):
        check = json.loads(path.read_text())
        if check.get("status") in {"ok", "failed"}:
            checks.append({"file": str(path.relative_to(ROOT)), **check})
    checks.sort(key=lambda c: (c["model"]["max_len"], c["model"].get("mixed_precision", "fp32"),
                               c.get("gradient_accumulation_steps", 1), c["questions_per_row"]))
    full = [c for c in checks if c["model"].get("encoder_last_n_layers") is None and c["questions_per_row"] == 1]
    fp32 = [c for c in full if c["model"].get("mixed_precision", "fp32") == "fp32"]
    def lengths(group, status, accumulated):
        return [c["model"]["max_len"] for c in group if c["status"] == status
                and (c.get("gradient_accumulation_steps", 1) > 1) == accumulated
                and (status != "ok" or len(c["steps"]) >= 2)]
    amp = [c for c in full if c["model"].get("mixed_precision", "fp32") != "fp32"]
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "protocol": "Synthetic full RLCD optimizer updates; forward precision and accumulation explicitly recorded; no dataset training or evaluation",
               "architectural_max_tokens": 8192, "mps_cap_gib": 9,
               "largest_tested_passing_context": max(lengths(fp32, "ok", False), default=None),
               "smallest_tested_failing_context": min(lengths(fp32, "failed", False), default=None),
               "largest_tested_passing_accumulated_context": max(lengths(fp32, "ok", True), default=None),
               "smallest_tested_failing_accumulated_context": min(lengths(fp32, "failed", True), default=None),
               "largest_tested_passing_amp_context": max(lengths(amp, "ok", True) + lengths(amp, "ok", False), default=None),
               "preset_tokens": 2048, "preset_precision": "fp16",
               "m1_encoder_last_n_layers": 2, "m1_accumulation_steps": 2, "checks": checks}
    (REPORTS / "finetune_memory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Unfrozen ModernBERT-large: context, mixed precision and checkpoints", "",
             "Full unfreezing trains all 421,029,889 encoder/head parameters. The M1 partial preset trains 50,763,777 parameters in the last two encoder blocks, final normalization and full head. Weights, gradients and Adam moments remain FP32 (6.27 GiB for full training, 2.14 GiB for partial training before activations). BF16/FP16 autocast changes suitable encoder and head forward operations. RLCD loss, probabilities and normalized entropy use FP32.", "",
             "| Tokens | Forward precision | Initial loss scale | Encoder blocks trained | Questions | Accumulation | Result | Observed MPS driver allocation | Peak process RSS | Seconds/update |",
             "| ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for c in checks:
        peak = c.get("measured_driver_peak_gib")
        seconds = sum(s["seconds"] for s in c["steps"]) / len(c["steps"]) if c["steps"] else None
        allocation = f"{peak:.2f} GiB" if c["status"] == "ok" and peak is not None else "Failed; see JSON error"
        timing = f"{seconds:.1f}" if c["status"] == "ok" and seconds is not None else "—"
        initial_scale = c["model"].get("fp16_initial_scale", 1024) if c["model"].get("mixed_precision") == "fp16" else "—"
        lines.append(f"| {c['model']['max_len']} | {c['model'].get('mixed_precision', 'fp32')} | {initial_scale} | {c['model'].get('encoder_last_n_layers') or 'All 28'} | {c['questions_per_row']} | {c.get('gradient_accumulation_steps', 1)} | {c['status']} | {allocation} | {c['process_peak_rss_gib']:.2f} GiB | {timing} |")
    lines += ["", "## Interpretation", "",
              "FP32 controls: 3,072 tokens passed with one row per update; 3,584 and 4,096 ran out of memory. With gradients retained across microbatches, 1,536 passed (including eight-row accumulation), while 2,048 and 3,072 ran out of memory. Those controls do not establish an AMP limit.", "",
              "The requested preset keeps 2,048 tokens, FP16 autocast with initial scale 1,024, one-row/one-question forwards, one row per optimizer update, encoder/head checkpointing, full RLCD, encoder LR 1e-5 and head LR 1e-4. See the AMP rows above for actual results. FP16 uses dynamic gradient scaling: unscale once after accumulation, clip, step, update the scaler and save its state for exact resume. BF16 does not require gradient scaling.", "",
              "The full-encoder five-question FP16 test also exceeded the cap at 2,048 tokens. The separate `finetune_large_m1` preset trains the last two encoder blocks and final normalization, plus the full head, with two-row accumulation. It retains the requested 2,048-token context and joint RLCD objective; its five-question, two-row accumulation preflight passed two updates at 5.03 GiB observed MPS driver allocation, with finite gradients and encoder weight updates. AMP evaluation forwards were finite. Full checkpoint save/resume passed: last.pt 1.95 GiB, inference best.pt 1.57 GiB; temporary files were removed.", "",
              "ModernBERT's architectural limit is 8,192 tokens. Context includes state, instructions, options and special tokens. BF16 and FP16 both ran out of memory with eight-row accumulation at 2,048 tokens. FP16 passed three one-row updates with a 1,024 initial loss scale, at 8.06 GiB observed driver allocation. The higher 65,536 initial scale caused nonfinite gradients in the one-row test. These are short synthetic memory checks, not convergence results or an exhaustive context maximum. Shapes, extra options and allocator reservation can change the limit. Driver allocation and process RSS are separate counters, not a measurement of total physical system RAM.", "",
              "A five-question FP32 row passed at 512 tokens, including full MPS save/resume: last.pt was 4.71 GiB and inference-only best.pt 1.57 GiB. Temporary test checkpoints were removed. Encoder gradients and weight updates were verified. Earlier frozen configs/checkpoints remain supported. Training metrics, loss scale and skipped optimizer updates are logged to Trackio.", "",
              "```sh", "uv run python train.py experiment=finetune_large_m1 device=mps", "# Reproduce AMP memory checks without reading any dataset/evaluation rows:",
              ".venv/bin/python scripts/preflight_finetune.py --max-len 2048 --mixed-precision fp16 --trackio", "```", "",
              "Sources: [Laya MPS recipe](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py), [Transformers memory accounting](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy), [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html), [MPS BF16 support](https://github.com/pytorch/pytorch/issues/139386).", ""]
    (REPORTS / "finetune_memory.md").write_text("\n".join(lines))
    print("Built finetune memory reports")


if __name__ == "__main__":
    main()
