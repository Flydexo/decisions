# Full-corpus time estimate on the 16 GiB M1, excluding MS MARCO

With the entire ModernBERT-large encoder trainable, a pre-norm transformer head, FP16, 1,024-token limit and full RLCD loss, plan **2–3 days per epoch for the four smoke datasets**, or roughly **6–22 weeks per epoch for the remaining 21-source collection, excluding MS MARCO**, running continuously. Three such epochs would therefore be roughly 18–66 weeks (about 4–15 months). These are extrapolations, not completed full-corpus runs or convergence estimates.

## Measured basis

The [completed smoke](finetune_1024_smoke.md) averaged 4.794 seconds per optimizer update of four one-question rows: **1.199 seconds per row**, approximately **3,004 rows/hour**. One million equivalent rows takes **13.9 days of training compute**, before validation, checkpointing, streaming and interruptions. The 1,024-token setting is a maximum; the smoke did not consist entirely of 1,024-token inputs.

| Scope | Expected eligible training workload | Compute-only extrapolation | Planning estimate, one epoch |
| --- | --- | --- | --- |
| AG News, BoolQ, SST-5, ARC-Challenge | About 125,181 rows after 10% validation reservation | 41.7 hours | 48–72 hours |
| Remaining 21 sources, excluding MS MARCO | Roughly 2.2–8 million equivalent rows, with substantial filtering uncertainty | About 31–111 days at smoke throughput | 42–154 days |

The four smoke sources contain 120,000 + 9,427 + 8,544 + 1,119 = **139,090 published training rows**. The validation partition is assigned by input hash, so the retained 90% count is an expectation, not an exact census.

## Where the remaining workload comes from

MS MARCO is excluded from this estimate as requested. Its metadata was already inspected and is retained in the JSON solely for provenance. CommitPackFT also remains excluded.

The pinned [Consumer Finance source](https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large/tree/44cfa170a402e254407470275ce05d7dcaccde30/data) contains **7,179,332 source rows**. The config excludes empty narratives and reserves 10% for final evaluation plus 10% for validation. At most approximately **5.74 million** remain for training before the narrative filter. Parquet null counts do not count empty strings, so eligible narratives were not exactly counted. This is a material uncertainty in the collection estimate. At the smoke rate, that upper eligible count alone adds approximately **80 compute days**. The 2.2–8 million range combines roughly 2.2 million equivalent rows from the other sources with zero to this upper limit; it is an uncertainty envelope, not a claim that the actual narrative count is zero or maximal.

The other known HF source splits contribute about **2.07 million source rows before holdouts**, excluding CodeReviewer, FlakeFlagger, the selected English support release and typed decisions. [CodeReviewer quality estimation](https://arxiv.org/pdf/2203.09095), Table 3, reports approximately 266,000 training examples. Aegis has two questions per accepted row; native typed decisions typically have five. These counts are not interchangeable with one-question model rows.

## Practical interpretation

This estimates one visit to every eligible training example, keeping validation and final evaluation separate. It does not specify how many epochs are needed for useful accuracy or calibration. Longer complaints, reviews and code, option counts, multi-question microbatching, thermal throttling and network I/O can change the smoke throughput substantially. The remaining mixture has not been benchmarked with full-encoder training.

Current presets use bounded training caches and step/time limits. A full-corpus run requires a separate streaming configuration that removes those sample and stopping limits, preserves split isolation, and verifies reader memory and archive transfer behavior. No new training was started for this estimate.

[Metadata and calculation inputs](training_time_estimate.json). Dataset metadata was read without downloading complete dataset files. Some small-source counts come from the current HF viewer because pinned cards lack split counts; that distinction is recorded in the JSON.
