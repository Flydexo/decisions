# Streaming MPS pilot — 5 October 2026

Implemented and exercised the Hydra training pipeline on a 16 GiB Apple M1. CommitPackFT was excluded as requested. All 17 requested dataset configs passed a live, one-row streaming schema check. These checks validate schema/access, not every row or label in each corpus.

## Training and memory

Final matched runs used 64 optimizer steps, batch size 2 (128 training rows), seed 42, a 2,048-row shuffle buffer per source, and round-robin AG News / BoolQ / SST-5 / ARC-Challenge. The encoder was frozen throughout; only 17,918,721 decision-head parameters trained. ModernBERT weights and HF dataset revisions are pinned in the configs.

| Measurement | Sampled reward | Cross-entropy ablation |
| --- | ---: | ---: |
| Live GPU tensor memory after updates | 0.757–0.757 GiB | 0.757–0.757 GiB |
| Driver allocation after updates | 1.015–2.074 GiB | 1.015–2.070 GiB |
| Mean step time after first four steps | 0.276 s | 0.276 s |
| Checkpoint including optimizer | approximately 205 MiB | approximately 205 MiB |

Live GPU tensor memory showed no growth across these runs. Driver allocations rose and fell as cached heaps were reused/cleared. These are post-update GPU measurements, not total unified-memory or peak system-RAM measurements; the short pilot does not establish long-duration leak freedom.

### Two concrete issues fixed

1. **Reward gradient direction.** The notebook differentiated a zero-centered Normal density through `logits + noise`, producing a gradient against the rewarded answer. The script uses detached sampled actions and evaluates their density under `Normal(logits, sigma)`. A regression test verifies that a rewarded answer becomes more probable after an update. Per-row advantage normalization and equal row weighting are preserved.
2. **Streaming order bias.** The first 32 AG News training rows sampled with a 32-row buffer were all Business. A 128-row buffer yielded only Business/Sci-Tech. At 2,048 rows the sample contained all four classes. The production default and final pilot use this larger bounded buffer. Bounded validation now shuffles too: AG News prefix accuracy of 75% dropped to 28.1% on the shuffled holdout sample. Prefix metrics overstated quality.

## Shuffled holdout evaluation

32 rows per dataset, seed 42, a bounded 2,048-row shuffle window, ordinary training-style truncation. Sampling from a bounded window still does not make this a uniform sample of the entire dataset. Both checkpoints use the same holdout rows.

| Dataset | Sampled reward accuracy | Cross-entropy accuracy | Sampled reward NLL | Cross-entropy NLL |
| --- | ---: | ---: | ---: | ---: |
| ag_news | 28.1% | 28.1% | 1.383 | 1.420 |
| boolq | 68.8% | 68.8% | 0.624 | 0.639 |
| sst5 | 25.0% | 25.0% | 1.606 | 1.562 |
| arc_challenge | 28.1% | 21.9% | 1.385 | 1.398 |

The small runs do not establish that one objective is better. Most multiclass predictions remain close to uniform; further training and broader holdout coverage are needed. Entropy confidence is a certainty measure; chosen-option probability is the quantity used for probability calibration.

## Decision Index source pilot

The final sampled-reward checkpoint was evaluated on the first 32 source-split rows from each source below, using the upstream direct-source request layout and full, untruncated payloads. The sample is independent of the frozen edition selection. ARC and MMLU are supplementary website tracks rather than evidence of coverage of every current index benchmark.

| Source | Answered / requested | Accuracy on supported rows | Median synchronized latency |
| --- | ---: | ---: | ---: |
| ARC-Easy | 32 / 32 | 21.9% | 38.9 ms |
| ARC-Challenge | 32 / 32 | 21.9% | 48.3 ms |
| MMLU | 32 / 32 | 15.6% | 44.8 ms |
| WinoGrande | 32 / 32 | 43.8% | 37.3 ms |
| HellaSwag | 32 / 32 | 21.9% | 47.4 ms |
| ANLI | 32 / 32 | 37.5% | 52.8 ms |
| BANKING77 | 0 / 32 | unsupported | — |

192 requests were answered, 32 were unsupported, and there were no source/runtime errors. Banking77 required 885–902 tokens in this sample with the complete option keys/descriptions, beyond the checkpoint’s 512-token capacity; no labels were removed or shortened to make it pass. These small source samples are not a full leaderboard evaluation and **no overall Decision Index score is claimed**.

The optional official reproduction kit is pinned to commit `87d4650b42b377c0291a89c1f1a879f9b31082bf` (edition 0.2.1). Its runner accepted the engine in an integration smoke check: two valid choice/Boolean requests and one correctly reported context overflow. The complete corpus is not publicly redistributed, and rebuilding it would require substantial dataset downloads, contrary to the requested bounded streaming pilot. The adapter and README include the full official evaluation/scoring path for an already-authorized local corpus. See the [official kit](https://github.com/apolinario/decision-index).

## Recommended next steps

- Keep BERT frozen, batch size 2, float32, bounded shuffling, and periodic cache release on the M1. Live tensor memory is stable in the pilot.
- Train substantially longer and increase holdout sample coverage before comparing model quality. Use shuffled validation; examine class balance, especially for rare flaky/safety labels.
- Run the configured component/reward ablations sequentially. `no_transformer` removes the largest trainable component and is the first speed/memory ablation to compare.
- For full large-option requests, train/configure a larger context capacity and start with batch size 1. Full benchmark payloads must remain intact.
- Save only the head/optimizer, and resume the deterministic stream. An interrupted partial optimizer update retains the previous atomic checkpoint rather than overwriting it.

## Reproduce / inspect

```sh
uv run python train.py experiment=pilot device=mps
uv run python train.py experiment=pilot ablation=cross_entropy device=mps
uv run python benchmark.py checkpoint="$PWD/outputs/pilot_streaming/last.pt" device=mps
uv run python -m unittest discover -s tests
```

All 22 offline tests passed, including exact CPU resume and interrupted-optimizer checkpoint protection. Local Trackio records were verified: no remote destinations or queued uploads.

- `reports/training_summary.json`: configs/environment, memory/time, diagnostic prefix metrics, shuffled holdout metrics.
- `reports/benchmark_summary.json`: source pins, coverage, latency, NLL/Brier, confidence, calibration diagnostics.
- `reports/schema_validation.json`: live schema checks for the 17 requested datasets.
- `outputs/pilot_streaming/last.pt`: final sampled-reward checkpoint.
- `outputs/pilot_streaming_cross_entropy/last.pt`: matched supervised ablation.
- `outputs/benchmark_streaming_pilot/results.jsonl`: per-request predictions and full payloads (kept locally).
