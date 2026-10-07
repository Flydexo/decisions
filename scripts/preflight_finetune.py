"""Measure full-encoder MPS training with synthetic inputs, without dataset/eval access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from decisions.evaluation import preprocessing
from decisions.checkpoints import read_checkpoint, resume_checkpoint, save_checkpoint
from decisions.losses import training_loss
from decisions.logging import Logger
from decisions.model import load_model, training_logits, inference_logits
from decisions.schema import collate, to_device
from decisions.trainer import build_optimizer, clear_cache, memory
from decisions.precision import build_scaler, optimizer_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--mixed-precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--questions", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-last-n-layers", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trackio", action="store_true", help="Log synthetic checks to decisions-finetune-large")
    parser.add_argument("--checkpoint-check", action="store_true",
                        help="Exercise full save/resume in a temporary directory; remove test checkpoints")
    args = parser.parse_args()
    if args.max_len < 32 or args.max_len > 8192 or args.questions < 1 or args.steps < 1 or args.accumulation_steps < 1:
        parser.error("Use 32–8192 tokens, positive question count and positive steps")
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="config", overrides=["experiment=finetune_large"]), resolve=True)
    cfg["model"]["max_len"] = args.max_len
    cfg["model"]["mixed_precision"] = args.mixed_precision
    cfg["model"]["encoder_last_n_layers"] = args.encoder_last_n_layers
    cfg["training"]["gradient_accumulation_steps"] = args.accumulation_steps
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run with GPU access")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.mps.set_per_process_memory_fraction(9 * 2**30 / torch.mps.recommended_max_memory())
    report = {"protocol": "Synthetic MPS training preflight; no dataset rows or final evaluation",
              "model": cfg["model"], "training": cfg["training"], "ablation": cfg["ablation"],
              "seed": 42, "torch_version": str(torch.__version__),
              "question_microbatch_size": 1, "batch_rows": 1,
              "gradient_accumulation_steps": args.accumulation_steps,
              "clear_cache_between_microbatches": cfg["training"].get("clear_cache_between_microbatches", False),
              "questions_per_row": args.questions, "mps_cap_gib": 9, "steps": []}
    variant = f"partial{args.encoder_last_n_layers}_" if args.encoder_last_n_layers is not None else ""
    if args.mixed_precision != "fp32":
        variant += f"{args.mixed_precision}_"
    if args.accumulation_steps != 1:
        variant += f"a{args.accumulation_steps}_"
    destination = args.out or ROOT / f"reports/finetune_preflight_{variant}{args.max_len}_q{args.questions}.json"
    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    logger = None
    try:
        if args.trackio:
            cfg["protocol"] = report["protocol"]
            logger = Logger(cfg["logging"], ROOT / "outputs" / destination.stem, cfg)
            report["trackio_project"] = cfg["logging"]["project"]
            report["trackio_run"] = destination.stem
        model, tokenizer = load_model(cfg["model"], cfg["ablation"], device)
        model.train()
        optimizer = build_optimizer(model, cfg["training"])
        scaler = build_scaler(cfg["model"], device)
        optimizer._decision_scaler = scaler
        trainable = [p for p in model.parameters() if p.requires_grad]
        report["total_parameters"] = sum(p.numel() for p in model.parameters())
        report["trainable_parameters"] = sum(p.numel() for p in trainable)
        report["encoder_trainable_parameters"] = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
        report["head_layer_checkpointing"] = model.head_checkpointing
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("AMP changed parameter storage dtype")
        report["fp32_weights_gradients_adam_gib"] = (
            4 * report["total_parameters"] + 12 * report["trainable_parameters"]) / 2**30
        print(json.dumps({k: v for k, v in report.items() if k not in {"model", "steps"}}), flush=True)
        question = {"type": "choice", "instructions": "Select the most appropriate option.",
                    "criteria": {"yes": "yes", "no": "no"}}
        row = {"state": "the " * args.max_len,
               "questions": {f"q{i}": question for i in range(args.questions)},
               "targets": {f"q{i}": {"yes": 1., "no": 0.} for i in range(args.questions)}}
        inputs, target, spans = collate([row], tokenizer, **preprocessing(cfg["model"]))
        inputs, target = to_device(inputs, device), target.to(device)
        report["sequence_length"] = inputs["input_ids"].shape[1]
        tracked = model.bert.final_norm.weight
        before = tracked.detach().clone()
        for step in range(args.steps):
            started = time.perf_counter()
            stages = []
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for micro in range(args.accumulation_steps):
                logits = training_logits(model, inputs, 1)
                logits_dtype = logits.dtype
                torch.mps.synchronize()
                stages.append({"stage": f"forward_{micro + 1}", **memory(device)})
                loss = training_loss(logits, target, inputs, spans, cfg["training"], cfg["ablation"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                scaler.scale(loss / args.accumulation_steps).backward()
                torch.mps.synchronize()
                stages.append({"stage": f"backward_{micro + 1}", **memory(device)})
                loss_value += loss.item() / args.accumulation_steps
                del logits, loss
                if cfg["training"].get("clear_cache_between_microbatches", False):
                    clear_cache(device)
                    stages.append({"stage": f"cache_clear_{micro + 1}", **memory(device)})
            encoder_grad_norm = tracked.grad.norm().item() / scaler.get_scale()
            if not encoder_grad_norm > 0:
                raise RuntimeError("Encoder gradient did not flow")
            norm, updated = optimizer_step(optimizer, trainable, cfg["training"]["clip_grad"], scaler)
            if not updated:
                raise RuntimeError("AMP skipped an optimizer update due to nonfinite gradients")
            torch.mps.synchronize()
            stages.append({"stage": "optimizer", **memory(device)})
            values = {"step": step + 1, "loss": loss_value, "seconds": time.perf_counter() - started,
                      "encoder_grad_norm": encoder_grad_norm, "gradient_norm": norm,
                      "loss_scale": scaler.get_scale(), "logits_dtype": str(logits_dtype), "memory_stages": stages}
            report["steps"].append(values)
            print(json.dumps(values), flush=True)
            if logger:
                logger.log({"step": step + 1, "train/loss": loss_value, "train/seconds": values["seconds"],
                            "train/encoder_grad_norm": encoder_grad_norm, "train/gradient_norm": norm,
                            "train/loss_scale": scaler.get_scale(), "context_tokens": args.max_len,
                            "accumulation_steps": args.accumulation_steps, **memory(device)}, step + 1)
            optimizer.zero_grad(set_to_none=True)
            if any(v.dtype != torch.float32 for state in optimizer.state.values()
                   for k, v in state.items() if k in {"exp_avg", "exp_avg_sq"}):
                raise RuntimeError("Adam moments are not FP32")
            clear_cache(device)
            save()
        model.eval()
        with torch.no_grad():
            evaluation_logits = inference_logits(model, inputs, 1).float()
            if not torch.isfinite(evaluation_logits).all():
                raise RuntimeError("Nonfinite evaluation logits after AMP training")
            report["evaluation_forward"] = "ok"
        del evaluation_logits
        clear_cache(device)
        report["encoder_weight_max_change"] = (tracked.detach() - before).abs().max().item()
        if args.checkpoint_check:
            cfg["dataset_configs"] = []
            print("Checking full encoder/optimizer checkpoint save and MPS resume...", flush=True)
            with tempfile.TemporaryDirectory(prefix="finetune-checkpoint-", dir=ROOT / "outputs") as directory:
                path = Path(directory) / "last.pt"
                save_checkpoint(path, model, optimizer, cfg, {"step": args.steps}, device)
                report["resume_checkpoint_gib"] = path.stat().st_size / 2**30
                expected = tracked.detach().cpu().clone()
                # A real resume starts with empty optimizer state; release the
                # old moments before measuring optimizer-state restoration.
                optimizer.state.clear()
                clear_cache(device)
                optimizer = build_optimizer(model, cfg["training"])
                optimizer._decision_scaler = scaler
                resume_checkpoint(path, model, optimizer, cfg, [], device)
                torch.testing.assert_close(tracked.detach().cpu(), expected, rtol=0, atol=0)
                report["checkpoint_resume_memory"] = memory(device)
                best = Path(directory) / "best.pt"
                save_checkpoint(best, model, optimizer, cfg, {"step": args.steps}, device, include_optimizer=False)
                saved = read_checkpoint(best)
                if saved["optimizer"] is not None or set(saved["encoder"]) != set(model.bert.state_dict()):
                    raise RuntimeError("Inference checkpoint omitted encoder weights")
                torch.testing.assert_close(saved["encoder"]["final_norm.weight"], expected, rtol=0, atol=0)
                report["inference_checkpoint_gib"] = best.stat().st_size / 2**30
                report["checkpoint_check"] = "ok"
                print(json.dumps({"checkpoint_check": "ok", "resume_checkpoint_gib": report["resume_checkpoint_gib"],
                                  "inference_checkpoint_gib": report["inference_checkpoint_gib"],
                                  "checkpoint_resume_memory": report["checkpoint_resume_memory"]}), flush=True)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failure_memory"] = memory(device)
        raise
    finally:
        report["process_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
        stages = [stage for step in report["steps"] for stage in step["memory_stages"]]
        if report.get("checkpoint_resume_memory"):
            stages.append(report["checkpoint_resume_memory"])
        report["measured_driver_peak_gib"] = max((s["memory/driver_gib"] for s in stages), default=None)
        report["measured_live_peak_gib"] = max((s["memory/live_gib"] for s in stages), default=None)
        save()
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
