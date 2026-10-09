# Decisions

Python training for the notebook's decision model: frozen or fine-tuned ModernBERT, Hydra
configuration, streaming datasets, local Trackio logging, checkpoints, ablations,
and normalized-entropy confidence.

## Run on an M1 Mac

```sh
uv sync
uv run python train.py experiment=pilot device=mps
uv run python train.py dataset=boolq training.batch_size=2 training.max_steps=1000 device=mps
```

Run from this directory; `main.py` also invokes training. `device=auto` selects
MPS, CUDA, then CPU. Explicit `device=mps` fails clearly if GPU access is missing.
Float32 and batch size 2 are the defaults for a 16 GiB M1. Batch size counts
**dataset rows**; a row with many questions consumes more memory.

Ordinary sources always use `load_dataset(..., streaming=True)`. Rows are
filtered, transformed, and batched as they arrive, using a bounded shuffle
buffer (2,048 rows by default). Bounded validation also shuffles its
stream to reduce bias from class-ordered prefixes. Labels come from config or ClassLabel metadata, without scanning a
source. Streaming transfers metadata and the chunks needed for the requested
rows; model weights are a separate one-time download. CodeReviewer uses a Hugging
Face `IterableDataset` over bounded HTTP ranges.

By default, BERT stays frozen and in evaluation mode, under `torch.no_grad()`. AdamW contains
only the decision layers. Gradients and temporary graphs are released after each
step; MPS cache is cleared periodically. `metrics.jsonl` records live tensor and
driver memory separately. Allocator reservation growth alone is not a tensor leak.

## Fine-tune ModernBERT-large

```sh
# Recommended M1 preset for the complete 22-source mixture:
uv run python train.py experiment=finetune_large_m1 device=mps
# Full unfreezing (single-question rows passed; five-question rows exceeded the cap):
uv run python train.py experiment=finetune_large device=mps
# Reproduce the synthetic memory check without reading any dataset/eval rows:
.venv/bin/python scripts/preflight_finetune.py --max-len 512 --questions 5
.venv/bin/python scripts/preflight_finetune.py --max-len 2048 --mixed-precision fp16 --trackio
```

`finetune_large` uses the 22-source streaming mixture and trains the entire encoder
and head with FP16 autocast. Weights, gradients and Adam moments stay FP32.
Encoder gradients stay connected; the encoder enters training
mode and AdamW includes its parameters. The head learning rate is `1e-4`, the
encoder rate is `1e-5`, and both follow the update-based schedule. The preset uses
2,048-token sequences, one-row updates, and
non-reentrant checkpointing of encoder and head layers. Multi-question rows also
use one-question checkpointed forwards: their logits are combined before the
original full RLCD reward is calculated, preserving its joint per-row loss.
The full-encoder five-question test exceeded the cap at 2,048 tokens.
`finetune_large_m1` trains the final two encoder blocks plus final normalization
and the complete head, with two-row accumulation and the same 2,048-token FP16
context. Use this separate preset for the complete mixture. Its two-update
five-question preflight passed at 5.03 GiB observed MPS driver allocation, with
finite gradients, encoder updates, and successful checkpoint save/resume.
Short final accumulation groups are normalized by their actual row count.
RLCD loss, probabilities and entropy use FP32 for numeric stability.
`model.mixed_precision=bf16` is also configurable; `fp32` disables autocast.
Autocast weight caching is disabled to reduce temporary copies.
FP16 uses dynamic loss scaling (initial scale 1,024); gradients are unscaled once
after accumulation and before clipping. Scaler
state is checkpointed; skipped updates and loss scale are logged to Trackio.
Autocast and FP16 training were verified on this Mac (PyTorch 2.14, macOS 26.5).
At 2,048 tokens, both FP16 and BF16 exceeded the cap with eight-row accumulation;
FP16 passed three one-row updates at 8.06 GiB observed driver allocation. Other
combinations need a preflight. See [the measured context report](reports/finetune_memory.md).

Before allocating model/optimizer memory, this preset prepares at most 4,000
training-only rows per source, streaming one source at a time and closing its
reader. Training then reads those bounded local pools; it avoids keeping 22
remote readers and shuffle buffers alive alongside the unfrozen model. Resume
uses the same run directory and validated sampling manifest.

The MPS allocation cap stays at 9 GiB; AdamW uses `foreach=false` to limit optimizer
temporaries. Shorter context reduces activations, but the 421,029,889 trainable
parameters still need about 6.27 GiB for FP32 weights, gradients, and Adam moments.
The architectural context limit is 8,192 tokens; the practical full-training limit
is measured separately in `reports/finetune_preflight_*.json`. Each context limit
includes instructions, options, state, and special tokens. These short synthetic
checks do not guarantee memory or convergence over a real long-running job.

This follows the encoder/head checkpointing approach in the
[Laya MPS recipe](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py),
with our own smaller microbatches and shorter context. Memory accounting follows
the [Transformers breakdown](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy).
`model.encoder_last_n_layers=N` offers partial unfreezing without adding a new
dependency. LoRA is not implemented by this preset.

Fine-tuned checkpoints include the encoder as well as the head; frozen-encoder
checkpoints remain readable. Full `last.pt` is roughly 4.7 GiB with Adam moments,
and inference-only `best.pt` roughly 1.57 GiB. Atomic replacement temporarily needs
space for another complete checkpoint, in addition to the 2 GiB disk reserve.
Serialization transfers GPU storages individually instead of copying the entire
checkpoint to CPU; reads use memory mapping. Resume verifies context, trainable
layers, accumulation, microbatch size, and learning rates. Use a new output
directory when switching from frozen training to this experiment.

Validation remains reserved from training; final evaluation does not select
checkpoints. Metrics, including the separate encoder learning rate, are logged
to local Trackio project `decisions-finetune-large`. The existing frozen configs
and saved experiments retain their behavior.

## Dataset configs

| Selection | Published source | Target / evaluation split |
| --- | --- | --- |
| `banking77` | `legacy-datasets/banking77` | Intent / test |
| `boolq` | `google/boolq` | Boolean `noul` / validation |
| `ag_news` | `fancyzhx/ag_news` | Topic / test |
| `mnli` | `nyu-mll/multi_nli` | NLI / validation_matched |
| `sst5` | `SetFit/sst5` | Ordered sentiment / test |
| `yelp_review_full` | `Yelp/yelp_review_full` | Ordered stars / test |
| `trec` | `CogComp/trec`, pinned Parquet export | Coarse question type / test |
| `dbpedia14` | `fancyzhx/dbpedia_14` | Entity type / test |
| `amazon_reviews_multi_en` | `SetFit/amazon_reviews_multi_en` | Ordered stars / test |
| `imdb` | `stanfordnlp/imdb` | Sentiment / test |
| `arc_challenge` | `allenai/ai2_arc`, ARC-Challenge | Variable MCQ options / test |
| `openbookqa` | `allenai/openbookqa`, main | Variable MCQ options / test |
| `commonsenseqa` | `tau/commonsense_qa` | Variable MCQ options / validation |
| `aegis` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | Prompt/response safety / test |
| `consumer_finance` | `davidheineman/consumer-finance-complaints-large` | Product routing / hash holdout |
| `codereviewer` | Microsoft CodeReviewer, [Zenodo 6900648](https://zenodo.org/records/6900648) | Comment on hunk / published test |
| `flakeflagger` | [Zenodo 5014076](https://zenodo.org/records/5014076) | Flaky from measured features / project holdout |
| `typed_decisions` | `LocalLLaMA/typed-decisions`, customer_service | Original native soft targets / test |
| `enron_spam` | `SetFit/enron_spam` | Spam Boolean / test |
| `phishing_email` | `zefang-liu/phishing-email-dataset` | Phishing Boolean / content hash holdout |
| `customer_support` | `Tobi-Bueck/customer-support-tickets` | English support queue / body hash holdout |
| `ms_marco` | `microsoft/ms_marco`, v2.1 | Query–passage selection Boolean / validation |
| `typed_decisions_all` | `LocalLLaMA/typed-decisions`, all | Four workflows with native soft targets / test |

Select with `dataset=<name>`. CommitPackFT is omitted as requested. TREC streams
CogComp's own pinned converter export because datasets 5 cannot load its old
script. CommonsenseQA uses validation because test labels are hidden. IMDb
unsupervised rows and empty finance narratives are excluded. Finance has an
explicit historical product vocabulary; unknown labels fail clearly.

The new email and support sources use pinned, explicitly selected JSON/CSV files
to avoid combining different releases. Support routing uses the newer bilingual
release, filtered to English; only the subject and body reach the model. Agent
answers, tags, priority, and the queue label stay out of its inputs. Phishing and
support reserve 10% for final evaluation by content hash. With
`data.separate_validation=true`, another 10% is reserved for checkpoint selection;
duplicate email/ticket bodies remain in one partition.

MS MARCO expands each query into one Boolean decision per passage, using
`passages.is_selected` as supervision. Multiple selected passages and all-negative
queries are preserved. Answer text and passage URLs are excluded from inputs.
Selection annotations serve as a relevance proxy; this config does not implement
the official passage-ranking benchmark. When validation is separated, every
passage for the same query stays in the same train/validation partition. Published
validation is reserved for final evaluation, since test annotations are hidden.
`schema.explode.fields` declares aligned lists to expand; `validation_group_field`
declares the source field used to group the validation holdout.

`typed_decisions_all` retains all native questions and probability targets from
the four-workflow `all` subset. Its test split remains untouched by training and
checkpoint selection. The existing customer-service-only config is preserved.

```sh
uv run python train.py dataset=enron_spam +data.separate_validation=true +training.validation_role=validation device=mps
uv run python train.py dataset=ms_marco +data.separate_validation=true +training.validation_role=validation device=mps
uv run python train.py experiment=expanded_large device=mps
.venv/bin/python scripts/validate_dataset_configs.py
```

`expanded_large` adds these five configs to the original 17-source mixture while
preserving the old experiment for checkpoint reproducibility. It uses the frozen
ModernBERT-large encoder, independently initialized pre-norm head, full RLCD
reward, and local Trackio project `decisions-expanded`. Batch size is one row for
both training and validation because native rows have five questions; the 9 GiB
MPS cap remains in place. Adding the configs does not launch training. Bounded
schema checks are saved in `reports/additional_dataset_validation.json` and are
separate from model performance measurements.

Aegis creates two safe/unsafe questions from the prompt/response pair, excluding
label and category columns from inputs. FlakeFlagger uses measured test features,
excluding its flaky label and identities. The holdout groups by project to avoid
project overlap. This is a feature-table task, not classification from source code.

CodeReviewer follows its [published labeling code](https://github.com/microsoft/CodeBERT/blob/master/CodeReviewer/code/utils.py):
nonempty `msg` means 1, otherwise use `y` or 0. Only `oldf` and `patch` are inputs.
Its reader fetches at most 256 KiB per range, requires exact HTTP 206 responses,
keeps one range block in memory, and caps each stream at 128 MiB transferred.
It never downloads or caches the full 2.8 GB archive. For longer runs, deliberately
increase `dataset.source.archive.max_transfer_bytes`.

## Map another dataset

Create `conf/dataset/my_dataset.yaml` and select `dataset=my_dataset`:

```yaml
name: my_dataset
source:
  path: owner/dataset
  name: default
  revision: PINNED_HUB_REVISION
splits:
  train: train
  eval: validation
schema:
  state:
    fields:
      context: passage
  questions:
    answer:
      type: choice
      instructions: "{question}"
      options:
        texts: choices.text
        keys: choices.label
      target:
        field: answerKey
        mode: label
```

Nested field paths are supported. State accepts `field`, `fields`, `template`, or
`literal`; instructions interpolate `{field.path}`. Questions support `choice`,
ordered `score`, and Boolean `noul`. Fixed options use `criteria` or
`labels_from_feature`. Targets support `index` with optional offset/mapping,
`label`, `boolean`, and soft `probabilities`. Native typed JSON columns use
`schema.native: true`. `derive` supports regex extraction/value maps; `filters`
supports nonempty/membership rules. `holdout` deterministically hashes an identity
or group field. Exclude target fields from state and instructions. JSON, CSV,
text, and Parquet URLs/local files use `source.format` and `source.data_files`.

Mixtures run in deterministic round robin, without materializing sources:

```sh
uv run python train.py 'mixture=[ag_news,boolq,sst5,arc_challenge]' training.max_steps=1000
```

Tail batches are retained. Rewards/advantages are normalized within each dataset
row, then row losses are averaged. Ordinal rewards use canonical level order even
when options are displayed in a different order. Training truncates context/option text to its
budgets while preserving every option marker. Large option sets need longer
context for faithful inference; strict benchmark mode rejects overflow instead
of truncating. Increase `model.max_len` cautiously on a 16 GiB machine.

## Logging, checkpoints, and ablations

Each Hydra run writes resolved config, per-step metrics, `last.pt`, `best.pt`, and
`evaluation.json` under `outputs/`. Trackio defaults to local storage in the
adjacent `trackio` directory; inherited remote destinations are ignored.
`logging.enabled=false` disables it. Open the local dashboard with:

```sh
TRACKIO_DIR="$PWD/outputs/trackio" uv run trackio show --project decisions
```

Checkpoints contain the trainable head, optimizer, configuration, RNG states, and
stream cursor. BERT reloads from its pinned revision. A baseline checkpoint is
about 205 MiB. Resume requires matching architecture, datasets, seed, batch size,
and optimizer settings; max_steps is the total target, not additional steps:

```sh
uv run python train.py experiment=pilot resume="$PWD/outputs/pilot_streaming/last.pt" training.max_steps=128
uv run python evaluate.py experiment=pilot resume="$PWD/outputs/pilot_streaming/last.pt"
uv run python train.py -m experiment=pilot ablation=baseline,no_question_type,no_transformer,linear_scorer,cross_entropy,no_spherical,no_rps
```

Resume reconstructs the deterministic stream and skips consumed canonical rows;
it may re-fetch earlier ranges, without materializing a dataset. Exact CPU replay
is tested; GPU determinism can differ. Atomic periodic checkpoints preserve
completed updates after abrupt termination. Hydra's basic multirun launcher
runs ablations sequentially, so multiple GPU models do not compete for memory.

The sampled-reward objective fixes a gradient-direction bug in the notebook:
samples are detached actions and log density is differentiated around trainable
logits. The old zero-centered density differentiated through samples moved logits
against the rewarded answer. The notebook remains an exploratory artifact.
Cross-entropy supports both hard and soft targets; reward ablations remove the
spherical or ranked-probability terms.

## Confidence

Every prediction returns the selected answer, all option probabilities, and
`chosen_probability`. The `confidence` field is the chosen option's probability
(after temperature scaling, if supplied), while `entropy_confidence` is:

```text
confidence = 1 + sum(p_k * log(max(p_k, 1e-12))) / log(K)
```

Only valid options count toward K. Uniform gives 0, one-hot gives 1, and a
singleton is defined as 1. Entropy confidence describes certainty; it is not a
probability of correctness. Evaluation reports accuracy, NLL, Brier,
chosen-option probability calibration error, and a diagnostic binned gap between
entropy certainty and accuracy. The entropy gap is not probability calibration.

## Decision Index

```sh
uv run python benchmark.py checkpoint="$PWD/outputs/pilot_streaming/last.pt" device=mps
```

This bounded pilot streams the first 32 source-split requests each from ARC-Easy,
ARC-Challenge, MMLU, WinoGrande, HellaSwag, ANLI r1, and Banking77. It mirrors the
upstream direct-source rendering and records synchronized latency, probabilities,
confidence, unsupported cases, and errors. These include supplementary tracks
listed on the website. The rows are independently sampled, not the frozen
leaderboard selection; **no overall Decision Index is reported for this pilot**.

### Decision Index 0.3: single-benchmark comparisons

The [official reproduction kit](https://github.com/apolinario/decision-index)
is pinned here to edition 0.3 at commit `9eb2dbe`. On **ARC-Easy**, the
released checkpoint answered all 2,376 requests and got **1,277 correct:
53.7% accuracy** (95% Wilson interval 51.7–55.7%). Uniform random choice
expects 25.0%. The [ARC-Easy comparison report](reports/bad-laya-arc-easy.json)
records the published reference scores and their source snapshot; the
[results page](reports/curriculum-results.html#index-section) visualizes them.
ARC-Easy is **shown on the public board but not counted in its overall index**.
The older [WinoGrande result](reports/bad-laya-winogrande.json) remains
642/1,267 = 50.7%, near its 50% random-choice expectation. WinoGrande is
counted in the index. Neither single-benchmark run establishes a full score.

The normalized ARC-Easy and WinoGrande sources and all-case selections match
the kit's official manifest. The full 0.3 suite is now hash verified, but
neither single-benchmark run is a full leaderboard submission. The kit's
`scores.json` records `complete: false`; its numerical `decision_index` field
is not meaningful for these partial runs and is not published as an index score.

With a locally imported 0.3 suite, extract ARC-Easy rows and run the released
checkpoint through the kit's strict engine on an Apple GPU:

```sh
uv sync --extra benchmark
uv run --extra benchmark python scripts/extract_decision_index_rows.py \
  outputs/decision_index_03/suite-0.3 26 2376 \
  outputs/decision_index_03/arc-easy-0.3.jsonl.gz
uv run --extra benchmark python -m decision_index run --edition 0.3 \
  --rows outputs/decision_index_03/arc-easy-0.3.jsonl.gz \
  --engine decisions.benchmark:DecisionEngine \
  --option checkpoint="$PWD/outputs/rtx4090_curriculum/full_split_ordered/last_bf16_fresh_optimizer.pt" \
  --option device=mps --option temperature=12.595144782442853 \
  --out outputs/decision_index_03/runs/bad-laya-arc-easy
uv run --extra benchmark python -m decision_index score --edition 0.3 \
  --suite-dir outputs/decision_index_03/suite-0.3 \
  --results outputs/decision_index_03/runs/bad-laya-arc-easy/results.jsonl \
  --engine decisions.benchmark:DecisionEngine \
  --out outputs/decision_index_03/runs/bad-laya-arc-easy
```

The full ARC-Easy inference took about 5.5 minutes on MPS. This command needs
an environment with Apple GPU access; the restricted shell can hide MPS even
when the host GPU is available. Benchmark questions and gold answers are not
committed. The run resumes from `results.jsonl`. Only `state` and `questions`
reach the model; gold/scoring fields stay with the evaluator. The model's
calibrated chosen-answer probability is `confidence`; normalized entropy
certainty is returned separately as `entropy_confidence`. Regenerate the
comparison with `scripts/build_arc_easy_comparison.py` and
`scripts/build_curriculum_results_page.py` once the kit score exists.

### Sequential full 0.3 evaluation (in progress)

`scripts/run_decision_index_03_sequential.py` runs all 43 public suite entries
one benchmark at a time on MPS, smaller counted benchmarks first. It writes
resumable rows and results under `outputs/decision_index_03/`, checks free
disk before each stage, and stops if a stage leaves pending requests or errors.
Its run directory is `outputs/decision_index_03/runs/bad-laya-full-03`.
To resume it in a shell with Apple GPU access:

```sh
.venv/bin/python -u scripts/run_decision_index_03_sequential.py
```

The pinned kit's RouterBench adapter uses ordinary `sum` for calibration
averages, while the released normalized files use compensated `math.fsum`.
`scripts/rebuild_canonical_decision_index_03.py` applies that correction to
RouterBench, verifies both published source hashes, rebuilds the suite, and
checks every official 0.3 row and exclusion hash. The
[canonical suite verification](reports/decision-index-03-canonical-suite.json)
records the hashes without redistributing benchmark inputs. An earlier
[audit of the provisional rebuild](reports/decision-index-03-suite-audit.json)
documents how the mismatch was isolated. The ongoing sequential run began
before the repair; its completed request payload hashes were checked against
the canonical suite, and its future RouterBench requests now use canonical
rows. The final results must be scored against `suite-0.3` with `complete:
true` before reporting a public index. The board's **Full score** also
includes private tests that only the maintainers run; this local process
computes the public component.

## Verification

### Small RLCD collapse pilots

```sh
.venv/bin/python scripts/run_collapse_pilots.py --run-dir outputs/collapse_rlcd_pilot
.venv/bin/python scripts/show_pilot_dashboard.py
```

The dashboard launcher selects the pilot database explicitly and opens a local
dashboard on port 7862. Select `prenorm_rlcd` and `no_transformer_rlcd` to compare
the runs. An already-open dashboard for `decisions` may use a different database
directory; open the URL printed by this launcher. Use `--check` to verify saved
runs without starting a server, or `--port` to choose another port.

The runner compares a pre-norm transformer with independently initialized layers
against a head without a transformer. Both use full sampled RLCD rewards
(log + 0.5 × spherical − ordinal RPS), frozen ModernBERT-large, batch size four,
32 candidates, and the same 256 streamed training examples across four datasets.
Each run has at most 256 updates and a 20-minute update/periodic-validation limit.
Terminal diagnostics and validation can add time afterward. Existing run
checkpoints are never overwritten; use a new run directory for repeats.

Trackio records loss, gradients, memory, separate validation metrics, and a fixed
eight-row training-only feature probe. The probe measures option RMS difference,
cosine similarity at each head layer, logit range, entropy confidence and each
reward component. Its reward is distinct from the stochastic training gradient
estimator loss. The aggregate report is `reports/collapse_rlcd_pilot_summary.json`;
the offline recap includes both trajectories when the pilots finish. No final
evaluation or Decision Index samples are used in these pilots.

### RTX 4090 24 GB pilot

[Instance setup, recovery and live Trackio instructions](docs/rtx4090_pilot.md)
cover the CUDA pilot. It trains the complete ModernBERT-large encoder with the
pre-norm transformer, BF16, 1,024 tokens and full RLCD. MS MARCO and CommitPackFT
are excluded; streaming pools are capped at 128 training rows per source.

```bash
# On a Linux RTX 4090 instance with NVIDIA driver >=580 and uv installed:
bash scripts/setup_rtx4090.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py
# Config inspection also works on the Mac; no training or downloads:
.venv/bin/python scripts/run_rtx4090_pilot.py --plan
```

The runner checks synthetic 1,024-token multi-question training and checkpoint
recovery before the pilot. It logs to `decisions-rtx4090-pilot`, records peak CUDA
memory and preserves an effective four-row batch when using smaller microbatches.
Local tests pass without CUDA; actual 4090 fit and speed are checked on launch.

### Full-encoder 1,024-token smoke

```sh
.venv/bin/python scripts/run_finetune_smoke.py
.venv/bin/python scripts/show_pilot_dashboard.py --check
```

This runs only the pre-norm transformer variant with the entire encoder unfrozen,
FP16 autocast, FP32 weights/Adam, encoder/head checkpointing, full RLCD and 32
candidates. One-row microbatches accumulate four rows per update. The smoke uses
64 updates / 256 presentations, compared to the frozen pre-norm and no-transformer
pilots **at step 64**, rather than their longer 256-update outcomes.

The runner verifies the exact existing training-pool and validation-sample hashes
and zero input overlap. It reuses the bounded HF-streamed samples without drawing
new final-evaluation data. It logs both encoder and head separation on the same
eight-row training probe, since the head/encoder ratio alone cannot detect encoder
collapse. Sampled early/late encoder weight changes are checked against the pinned
pretrained weights. No final test split or Decision Index source is evaluated.

Trackio uses your existing `decisions-collapse-rlcd` database and the run
`finetune_prenorm_rlcd_1024_smoke`. Refresh the live dashboard to select it.
The aggregate measurements are in `reports/finetune_1024_smoke_summary.json`;
[the smoke report](reports/finetune_1024_smoke.md) and [recap](reports/results.html)
show matched comparisons. For an interrupted run, confirm no matching worker is
alive before `scripts/run_finetune_smoke.py --resume`; the time budget accumulates
across recovery. Existing frozen pilots are never overwritten.

The completed ModernBERT-large run is documented in [the offline recap](reports/results.html)
and [its audit](reports/all_large_audit.json). It processed 20,520 rows across all
17 datasets; validation selected update 3,000. All 224 bounded benchmark source
requests were answered, but the official Decision Index was not computed.
Probabilities stayed nearly uniform, and the FlakeFlagger final sample contained
256 non-flaky tests and no flaky tests. These results do not establish useful learning.
An [inference-only head diagnostic](reports/head_inference_diagnostic.json) on two
training examples found option-marker cosine similarity rising from 0.9300 in
the encoder to 1.0000 after the head transformer. CPU and MPS probabilities
agreed within 7.45e-8; this small check points to head representation collapse.

The audit found duplicate Consumer Finance narratives under different complaint
IDs: 8 validation rows and 10 final rows matched inputs seen at the selected
checkpoint. This affects the complaint scores and checkpoint selection. Future
configs partition by narrative text; existing checkpoints, scores and source
provenance retain the old ID-based split. Rebuild caches for a new run.
The update timer stopped at eight hours, followed by 13.8 minutes of terminal
validation; final evaluation and benchmark sampling ran afterward. Recorded
GPU driver allocation peaked at 8.09 GiB, with live tensors flat at 1.77 GiB.

```sh
uv run python -m unittest discover -s tests
```

Tests cover streaming laziness, schema adapters, option permutations/padding,
entropy confidence, sampled gradient direction, frozen encoder ablations,
transfer limits, tail batches, exact CPU resume, and interruption recovery. Recorded MPS pilot and
benchmark results are in `reports/pilot.md`.

Open `reports/results.html` for the offline interactive recap: objective comparisons,
GPU memory and timing traces, sampling diagnostics, benchmark coverage, and searchable
dataset schema mappings. The page includes CSV/JSON exports and marks the five unrun
ablations explicitly. It has no external assets, telemetry, or live dataset requests.

Rebuild it from the saved measurements with:

```sh
uv run python scripts/build_results_page.py
```

The [RTX 4090 curriculum results page](reports/curriculum-results.html) compares
the locally evaluated partial stage-16 checkpoint with uniform random guessing,
the RTX 4090 pilot baseline, and the last completed curriculum stage. It also
compares accuracy with entropy confidence and reports chosen-probability
calibration error. It includes all 21 validation datasets and opens offline.
Rebuild the page from its tracked data snapshot with
`python3 scripts/build_curriculum_results_page.py`; use
`--refresh` to recompute that snapshot from the local checkpoint evaluation and
validation samples.

The partial stage-16 model is published as
[bad-laya on Hugging Face](https://huggingface.co/flydexo/bad-laya), with BF16
encoder weights, FP32 decision-head weights, tokenizer files, and an evaluation
model card. Recreate the upload folder from the verified local checkpoint with
`.venv/bin/python scripts/build_bad_laya_card_art.py` and
`.venv/bin/python scripts/export_bad_laya_hf.py`.

For ModernBERT-large across all 17 requested datasets:

```sh
uv run python scripts/run_all_large.py --hours 8 --run-dir outputs/all_large_8h
```

This run keeps the encoder frozen, uses 1,024-token context, preserves complete
options when they fit, and shares truncated structured context across its fields.
It streams and caches at most 4,000 **training-only** rows per dataset, preparing one
source at a time to avoid retaining 17 remote readers and shuffle buffers in RAM.
It then trains the balanced mixture from those bounded pools with cross-entropy,
four-row batches, learning-rate warm-up and cosine decay. GPU allocation is capped
at 9 GiB, targeting about 10 GiB including process RAM. The synthetic maximum-length
four-row/two-question preflight used 8.61 GiB driver memory and 0.60 GiB peak process
RSS; actual memory is logged throughout training. This samples every dataset; it is not a
full-corpus training run.

Checkpoint selection uses 128 validation rows per dataset. Validation is reserved
from training by a hash of the inputs; Finance and FlakeFlagger use disjoint
identity/project hash regions; CodeReviewer uses its official validation split.
The original evaluation splits are never used for checkpoint selection. After
training, the script evaluates `best.pt` once on up to 256 rows per original
evaluation split and writes `reports/all_large_summary.json`.

`last.pt` includes the optimizer for exact resume; `best.pt` contains the head for
inference only, reducing disk usage. Resume with the same run directory:

```sh
uv run python scripts/run_all_large.py --hours 8 --run-dir outputs/all_large_8h --resume outputs/all_large_8h/last.pt
```

The training budget accumulates across resumes.
DBpedia 14, Amazon Reviews and IMDb are class-ordered. Their bounded training,
validation and final samples use equal class quotas from label-filtered Parquet
streams, followed by training-pool shuffling. Other sources keep their existing
bounded sampling. This prevents a small shuffle buffer from selecting just the
first class; their reported accuracy measures a class-balanced sample rather
than the original frequency distribution. Training and validation still use
disjoint input hashes; final samples come only from the original evaluation split.
The initial single-class control is archived and excluded from checkpoint selection.
`reports/sampling_correction.json` records its exposure and consumed budget.

When restarting a corrected experiment from a fresh head, `--budget-spent-seconds`
deducts archived training/validation time from the total `--hours` allowance.
The time limit applies to training including periodic validation; source preparation
and final evaluation add time. Training also stops after four validation checks
without improved mean accuracy or if the disk reserve is reached.
