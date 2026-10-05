# Decisions

Python training for the notebook's decision model: frozen ModernBERT, Hydra
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

BERT stays frozen and in evaluation mode, under `torch.no_grad()`. AdamW contains
only the decision layers. Gradients and temporary graphs are released after each
step; MPS cache is cleared periodically. `metrics.jsonl` records live tensor and
driver memory separately. Allocator reservation growth alone is not a tensor leak.

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

Select with `dataset=<name>`. CommitPackFT is omitted as requested. TREC streams
CogComp's own pinned converter export because datasets 5 cannot load its old
script. CommonsenseQA uses validation because test labels are hidden. IMDb
unsupervised rows and empty finance narratives are excluded. Finance has an
explicit historical product vocabulary; unknown labels fail clearly.

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

Every prediction returns the selected answer, all option probabilities,
`chosen_probability`, and:

```text
confidence = 1 + sum(p_k * log(max(p_k, 1e-12))) / log(K)
```

Only valid options count toward K. Uniform gives 0, one-hot gives 1, and a
singleton is defined as 1. Entropy confidence describes certainty; it is not a
calibrated probability of correctness. Evaluation reports accuracy, NLL, Brier,
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

The [official reproduction kit](https://github.com/apolinario/decision-index)
uses edition 0.2.1. Its complete frozen corpus is not publicly redistributed;
rebuilding requires substantial downloads and working space. This pipeline does
not trigger that rebuild. When an authorized local frozen suite is available,
the pinned optional kit can run our strict engine and its official scorers:

```sh
uv sync --extra benchmark
uv run --extra benchmark python -m decision_index run \
  --edition 0.2.1 --engine decisions.benchmark:DecisionEngine \
  --option checkpoint="$PWD/outputs/pilot_streaming/last.pt" --option device=mps \
  --rows /absolute/path/to/selected-rows.jsonl.gz --out outputs/full-index
uv run --extra benchmark python -m decision_index run \
  --edition 0.2.1 --engine decisions.benchmark:DecisionEngine \
  --option checkpoint="$PWD/outputs/pilot_streaming/last.pt" --option device=mps \
  --rows /absolute/path/to/added-rows.jsonl.gz --out outputs/full-index
uv run --extra benchmark python -m decision_index score --edition 0.2.1 \
  --suite-dir /absolute/path/to/verified-suite \
  --results outputs/full-index/results.jsonl --out outputs/full-index
```

These commands read supplied row files incrementally; they do not download or
rebuild a suite. Only `state` and `questions` reach the model. Gold/scoring fields
stay with the evaluator. No option pruning or context truncation is allowed;
requests beyond capacity become `unsupported`. The official kit supplies
benchmark-specific scoring, coverage, edition selection, and the leaderboard
formula. Its calibration uses the returned probabilities; entropy confidence
is retained separately.

## Verification

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
