# Full-encoder ModernBERT-large smoke at 1,024 tokens

One full-encoder 1024-token FP16 pre-norm RLCD smoke; saved frozen baselines compared at 64 updates / 256 row presentations. No new no-transformer run, final evaluation or benchmark samples.

## Matched comparison

All three rows below are measured at **64 optimizer-update attempts and 256 training-row presentations**. The frozen baselines are saved observations from the previous pilots; only the whole-encoder pre-norm variant was trained in this smoke.

| Variant | Mean validation accuracy | Encoder option RMS | Head option RMS | Encoder cosine | Head cosine | Probe RLCD reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen · prenorm_rlcd | 30.5% | 0.3059 | 0.3072 | 0.9027 | 0.9762 | -0.9868 |
| Frozen · no_transformer_rlcd | 28.1% | 0.3059 | 0.3059 | 0.9027 | 0.9434 | -1.0682 |
| Whole encoder · pre-norm RLCD | 35.2% | 0.4467 | 0.5996 | 0.6615 | 0.7384 | -2.1130 |

Datasets with higher validation accuracy than frozen pre-norm: sst5, arc_challenge. Calibration worsens: mean validation NLL is **3.05**, versus **1.29** for frozen pre-norm. BoolQ entropy confidence is 100.000% at 59.4% accuracy, matching the always-true baseline for this 19-true/13-false sample. Preventing feature collapse does not establish reliable decisions.

| Dataset | Frozen pre-norm | Frozen no transformer | Whole encoder + pre-norm |
| --- | ---: | ---: | ---: |
| ag_news | 21.9% | 21.9% | 21.9% |
| boolq | 59.4% | 59.4% | 59.4% |
| sst5 | 15.6% | 9.4% | 21.9% |
| arc_challenge | 25.0% | 21.9% | 37.5% |

## Calibration on the same validation rows

| Variant / dataset | NLL | Brier | Mean entropy confidence | Probability ECE |
| --- | ---: | ---: | ---: | ---: |
| Frozen · prenorm_rlcd / ag_news | 1.4066 | 0.7616 | 1.9% | 0.1192 |
| Frozen · prenorm_rlcd / boolq | 0.7508 | 0.5430 | 21.0% | 0.1677 |
| Frozen · prenorm_rlcd / sst5 | 1.6192 | 0.8039 | 0.0% | 0.0534 |
| Frozen · prenorm_rlcd / arc_challenge | 1.3732 | 0.7413 | 2.1% | 0.1181 |
| Frozen · no_transformer_rlcd / ag_news | 1.4491 | 0.7892 | 4.1% | 0.1810 |
| Frozen · no_transformer_rlcd / boolq | 1.0052 | 0.6704 | 52.5% | 0.3031 |
| Frozen · no_transformer_rlcd / sst5 | 1.6329 | 0.8083 | 0.5% | 0.1271 |
| Frozen · no_transformer_rlcd / arc_challenge | 1.4336 | 0.7691 | 4.2% | 0.1220 |
| Whole encoder · pre-norm RLCD / ag_news | 2.7734 | 1.3687 | 73.3% | 0.6929 |
| Whole encoder · pre-norm RLCD / boolq | 6.3892 | 0.8125 | 100.0% | 0.4062 |
| Whole encoder · pre-norm RLCD / sst5 | 1.6067 | 0.7990 | 0.0% | 0.0095 |
| Whole encoder · pre-norm RLCD / arc_challenge | 1.4114 | 0.7605 | 0.8% | 0.0983 |

## Stability and actual encoder updates

- All 421,029,889 encoder/head parameters trainable; FP16 forward operations with FP32 parameter, gradient and Adam storage.
- Full RLCD: log + 0.5 × spherical − ordinal RPS, with 32 candidates and sigma 1. No CE-only substitution.
- One-row microbatches, four-row accumulation, non-reentrant encoder/head checkpointing; encoder LR 1e-5 and head LR 1e-4.
- 64 successful updates, 0 skipped by gradient scaling.
- Encoder option RMS on eight fixed training examples: 0.3059 → 0.4467 (1.460 × initial).
- Head/encoder RMS ratio at the end: 1.3425; head cosine: 0.7384.
- Head-collapse rule flagged: False; encoder-collapse rule flagged: False.
- Probe reward: -1.0439 → -2.1130. Stochastic RLCD gradient-estimator loss is not CE and does not rank model quality.

The head-collapse rule checks a head/encoder RMS ratio below 0.01 and head cosine above 0.999. The encoder rule checks encoder RMS below 0.01 × its initial value and encoder cosine above 0.999. These are diagnostics on the fixed probe, not general guarantees.

Early and late sampled parameter changes against the pinned pretrained encoder:

- `layers.0.attn.Wqkv.weight`: max absolute change 0.00026280247.
- `layers.27.mlp.Wi.weight`: max absolute change 0.00025465712.
- `final_norm.weight`: max absolute change 0.00022089481.

## Resources and logging

- MPS allocation cap: 9 GiB. Peak logged driver allocation: 7.06 GiB.
- Post-update live allocations: 4.716–4.716 GiB. Peak process RSS: 0.84 GiB.
- Mean update time: 4.79 seconds (periodic validation and checkpoint serialization excluded).
- These separate counters do not measure total physical RAM or internal kernel peaks.
- Trackio project `decisions-collapse-rlcd`, run `finetune_prenorm_rlcd_1024_smoke`.
- Checkpoints: last.pt 4.71 GiB, best.pt 1.57 GiB.

## Sampling and limitations

The same bounded HF-streamed training pools and 32 validation rows per source were reused and checked by SHA-256. Training/validation input overlaps are zero. The first epoch follows the same deterministic row order as the frozen pilots. No final test split or Decision Index request was read.

- One seed; 256 unique training rows and 128 reused validation rows across four sources.
- FP16, microbatching and encoder learning rate differ from the original frozen FP32 pilots; this is not an isolated unfreezing ablation.
- A short stability/learning smoke does not establish convergence or generalization.
- RMS collapse rules are diagnostics, not guarantees. Encoder and head collapse are checked separately.
- Logged memory snapshots are not a measurement of total physical RAM or internal kernel peaks.

The longer frozen pilots ran 256 updates / 1,024 presentations; their terminal scores are not the matched comparison above. Validation checkpoint selection is separate from final evaluation, which was not run.

```sh
.venv/bin/python scripts/run_finetune_smoke.py
# Exact recovery after confirming no matching GPU worker is alive:
.venv/bin/python scripts/run_finetune_smoke.py --resume
.venv/bin/python scripts/show_pilot_dashboard.py --check
```

[Interactive recap](results.html) · [All aggregate measurements](finetune_1024_smoke_summary.json)
