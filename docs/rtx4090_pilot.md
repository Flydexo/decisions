# RTX 4090 24 GB pilot on Vast.ai

One pre-norm transformer pilot: full pinned ModernBERT-large encoder, 1,024 tokens, independent head initialization, full RLCD (log + 0.5 spherical − ordinal RPS), 32 candidates and BF16 autocast. Weights, gradients and Adam moments remain FP32. MS MARCO and CommitPackFT are excluded.

The default uses at most **128 training-only rows from each of 21 sources**: at most 2,688 rows, one epoch, 672 updates and a 1,200-second training budget including periodic validation. Source preparation, synthetic preflight and terminal validation take additional time. No final test split or official Decision Index evaluation is run.

## Instance and install

Choose one full RTX 4090 24 GB, on-demand, a verified host with preferably >=99.5% reliability, 64 GB host RAM, 8–16 allocated CPU cores, 100 GB allocated NVMe/SSD and at least 500 Mbps download. Check bandwidth prices because HF streaming still transfers data.

Use a Linux x86_64 Ubuntu 22.04/24.04 or compatible Python/PyTorch SSH template, with **NVIDIA driver >=580**. The frozen lock installs PyTorch 2.14, Transformers 5.17 and CUDA 13.0.3 into `.venv`; an old CUDA 12-only host is insufficient. [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

On the rented instance:

```bash
git clone https://github.com/Flydexo/decisions.git
cd decisions
python3 -m pip install uv
bash scripts/setup_rtx4090.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py
```

Setup runs `uv sync --frozen --python 3.11`, leaving the template's global PyTorch untouched. It does not train. The runner checks the actual CUDA runtime, BF16, exactly one visible 4090, at least 21 GiB currently free GPU memory and 20 GiB free disk. The PyTorch allocator cap is **21 GiB**, leaving headroom; it does not cap all device/driver memory. Public sources normally require no HF token. No CPU/MPS fallback is allowed.

## What the runner does

First it runs four real GPU optimizer updates on synthetic full-length inputs: four rows, five questions each, 77 options on one question and **1,024 actual tokens per question**. Question forwards are limited to four at a time while preserving the joint RLCD reward per row. It checks early/late encoder gradients and weight changes, FP32 Adam states, and full checkpoint/RNG recovery. The temporary checkpoint is removed. Trackio logs this separately under a run ending in `__preflight`.

The synthetic model is released; the pilot starts from pretrained encoder weights and a fresh head. HF sources use `streaming=True`. Only bounded training pools and validation samples are cached, with readers prepared one source at a time before model allocation. Revisions, schemas, partitions and pool hashes are saved.

Four-row batches use one accumulation step. Encoder/head checkpointing remains enabled. Per-microbatch cache clearing is disabled; validation still releases caches. Existing SDPA attention is retained. FlashAttention, larger batches and reduced checkpointing should be benchmarked separately after this baseline passes. A 16-update learning-rate warmup is enabled; encoder LR is 1e-5, head LR 1e-4.

For a separate throughput run after the baseline, `--batch-size 8` or `16` uses one update per batch, and `--question-microbatch-size 8`, `16`, or `20` increases the number of question forwards per chunk. Use a fresh `--run-dir`; larger row batches change the optimization schedule. Compare synchronized `train/seconds` per row and CUDA peak memory, and keep the setting below the 21 GiB allocator cap.

## Checks and a smaller microbatch

```bash
# Any machine: config only, no GPU/model/data access or output directory creation.
.venv/bin/python scripts/run_rtx4090_pilot.py --plan

# Instance: environment checks only, no model/data downloads.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py --check-env

# Instance: synthetic GPU training and checkpoint recovery only.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py --preflight-only

# If the default preflight hits OOM, use a fresh directory and smaller chunks.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py \
  --batch-size 1 --question-microbatch-size 1 \
  --run-dir outputs/rtx4090_pilot/prenorm_bf16_b1q1
```

One-row batches accumulate four times, preserving effective batch size four. Two-row batches accumulate twice. The runner does not silently change settings after OOM. Synthetic preflight is a fit check for its cases, not a guarantee for every source. A directory lock rejects duplicate workers launched through this runner.

## Recovery and results

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py --resume

# Lower-memory variant must resume with identical flags.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py \
  --batch-size 1 --question-microbatch-size 1 \
  --run-dir outputs/rtx4090_pilot/prenorm_bf16_b1q1 --resume
```

Stop the original worker before resuming. Fresh launches reject existing checkpoints. Resume skips the synthetic preflight, verifies model/optimizer/batch/pool/partition settings and restores CUDA RNG. The training timer accumulates. Interrupted updates retain the last atomic checkpoint. MPS checkpoint resume is rejected; keep the same CUDA run-directory path because absolute cache provenance is checked.

Default artifacts in `outputs/rtx4090_pilot/prenorm_bf16_b4/`:

- `environment.json`, `preflight.json`, `status.json`, `config.json`.
- `metrics.jsonl`: synchronized update timing, gradient norm, learning rates, optimizer success and peak allocated/reserved CUDA memory per update.
- `feature_diagnostics.jsonl`: fixed training-only encoder/head RMS and cosine, RLCD reward and calibration.
- `validation.json`, `validation_history.jsonl`: up to 32 validation rows per source, isolated from training and final evaluation.
- `last.pt`: full encoder/head/Adam/RNG recovery (~4.7 GiB after Adam initialization); `best.pt`: validation-selected model (~1.6 GiB) without optimizer.
- `summary.json`: actual source exposure, pool hashes, input-overlap audit, selected validation scores, feature diagnostics and timing.

Checkpoint selection uses mean per-source validation accuracy. NLL/Brier/ECE reveal calibration problems. The new mixture, BF16 and warmup differ from the M1 smoke: this is not an isolated hardware ablation. CUDA update peaks exclude separate validation/diagnostic windows; update timing excludes periodic checkpointing/validation. Pools, checkpoints and credentials remain ignored by Git.

## Live Trackio

On the instance, in a second terminal:

```bash
.venv/bin/python scripts/show_pilot_dashboard.py \
  --run-dir outputs/rtx4090_pilot --project decisions-rtx4090-pilot \
  --port 7862 --no-browser
```

From the Mac, using Vast's SSH host/port:

```bash
ssh -N -L 7863:127.0.0.1:7862 -p YOUR_SSH_PORT root@YOUR_HOST
```

Open **http://127.0.0.1:7863**. Select project **`decisions-rtx4090-pilot`**, run **`prenorm_bf16_b4`**. Port 7863 avoids the existing Mac dashboard on 7862. The remote dashboard binds to loopback; no public dashboard port is needed. Add `--check` to verify saved runs without starting a server.

## Validation status

```bash
.venv/bin/python -m unittest discover -s tests
```

Preparation was tested on the M1: config, synthetic schema, memory/hardware checks, worker locking and existing CPU BF16/gradient/checkpoint behavior. **The actual RTX 4090 GPU preflight and pilot have not been executed here.** No instance was rented or billed.
