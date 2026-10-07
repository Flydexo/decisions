# Full RLCD feature-collapse pilots

Matched small pilots; fresh frozen ModernBERT-large, identical training pools and validation partitions. No final evaluation or benchmark tuning.

Both requested pilots completed 256 updates with batch size four (1,024 row presentations each). Each saw the same 256 unique training rows: 64 per dataset. Encoder: frozen ModernBERT-large; reward: log + 0.5 spherical − ordinal RPS; 32 candidates, sigma 1.0, learning rate 0.0001.

| Pilot | Final option spread / encoder spread | Final option cosine | Final score range | Selected validation accuracy | Selected step | Peak recorded driver GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-norm + independent initialization | 1.4332 | 0.943922 | 3.4113 | 39.06% | 256 | 3.12 |
| No transformer | 1.0000 | 0.943319 | 2.9803 | 34.38% | 192 | 2.03 |

No feature collapse was detected on the fixed eight-row training probe in either run. The old overnight model had nearly identical option features and nearly uniform scores. Here the final scores are clearly distinct. These runs jointly change architecture, initialization and objective relative to that older experiment; they do not isolate which change prevented its failure.

The no-transformer representation has a constant raw option spread by construction: the encoder is frozen and the question-type offset is identical for all options. That does not guarantee its scorer learns useful distinctions. The pre-norm head retains a direct residual route for encoder features. Higher spread means greater numerical differences, not necessarily better semantic information.

## Learning quality remains unresolved

| Pilot | Probe NLL, start → final | Probe full reward, start → final | Selected BoolQ accuracy | Selected BoolQ NLL | Selected BoolQ entropy confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| prenorm_rlcd | 1.284 → 2.083 | -1.044 → -1.834 | 59.38% | 4.819 | 99.97% |
| no_transformer_rlcd | 1.249 → 1.888 | -0.999 → -1.622 | 59.38% | 2.635 | 97.61% |

Both selected models get 19/32 BoolQ validation answers right, exactly the accuracy of always answering true in this validation sample. Their high certainty and high NLL remain a concern. The fixed-probe reward became worse despite some gains in probe accuracy, showing why gradient-estimator loss and accuracy alone are insufficient learning-quality measures. No probability recalibration, new tuning or further training was performed after these observations.

Validation contains the same 32 rows per dataset in both runs; canonical cache SHA-256 values match. Training/validation input overlaps are zero. No original final evaluation split or Decision Index samples were read. All class histograms and trajectories are in the JSON summary.

These are small, one-seed experiments. The selected accuracies differ by only six correct answers across 128 validation questions. Common scorer weights are not forced identical between the different architectures; their constructors consume RNG differently. A clean training-only overfit test and further controlled experiments are still needed before scaling up.

## Trackio

Both runs are persisted under project `decisions-collapse-rlcd` with names `prenorm_rlcd` and `no_transformer_rlcd`. Database verification confirmed all nine feature probes per run (steps 0–256), training metrics and dataset validation scores.

```sh
TRACKIO_DIR="$PWD/outputs/collapse_rlcd_pilot/trackio" .venv/bin/trackio show --project decisions-collapse-rlcd
```

Open the dashboard from the project directory. The run directories retain best and last head checkpoints; they are excluded from Git. The offline recap contains interactive feature and validation charts.
