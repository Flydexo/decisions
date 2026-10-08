---
language: en
base_model: answerdotai/ModernBERT-large
library_name: pytorch
metrics:
  - accuracy
tags:
  - decision-model
  - multiple-choice
  - modernbert
  - experimental
  - bf16
thumbnail: https://huggingface.co/flydexo/bad-laya/resolve/main/assets/overview.png
---

# bad-laya

**An open-weight research checkpoint for typed decisions.** Give it a state and a question with explicit options; its ModernBERT encoder and decision head score those options without generating text. This is the **partial stage-16 checkpoint** of an ordered curriculum, not the best checkpoint of the run.

![Evaluation overview](https://huggingface.co/flydexo/bad-laya/resolve/main/assets/overview.png)

**Explore:** [Results](#at-a-glance) · [Quickstart](#load-and-score-a-request) · [Calibration](#temperature-calibration) · [Provenance](#training-and-evaluation-provenance) · [Limitations](#limitations-and-use)

> **Research use.** The model is weak on some review tasks, even after calibration. The 21-source comparison uses validation samples; the calibration check uses separate evaluation partitions. An official Decision Index 0.3 score has not yet been completed.

## Model details

| | |
|:--|:--|
| Developer | [flydexo](https://huggingface.co/flydexo) |
| Backbone | Fine-tuned `answerdotai/ModernBERT-large` at pinned revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` |
| Decision head | Two transformer layers with one learned score per supplied option marker |
| Input | A text or JSON state, plus named `choice`, `score`, or `noul` questions |
| Output | A probability for each option; selected choice, expected score, or probability of true |
| Context | Up to 1,024 tokens per question under the saved preprocessing configuration |
| Weights | 895 MB Safetensors: BF16 encoder and FP32 decision head |
| Calibration | One post-training temperature, **T = 12.60**, stored in `calibration.json` |
| License | Not declared for this combined checkpoint; the ModernBERT base is Apache 2.0 |

The question’s options define the answer space at inference time. `choice` returns the most likely named option, `score` returns the probability-weighted level index, and `noul` returns the probability of true. The custom [`decisions` code](https://github.com/Flydexo/decisions) handles preprocessing and option-marker scoring. **This repository is not a drop-in `transformers.AutoModel.from_pretrained` or text-generation model.**

## At a glance

| 21-source validation macro average | Score |
|:--|--:|
| **bad-laya · step 60,928** | **65.2% accuracy** |
| Uniform random choice over each question's valid options | 29.0% expected accuracy |
| Earlier bounded RTX 4090 pilot | 43.3% accuracy |
| Previous complete curriculum stage (stage 15) | 67.3% accuracy |
| Normalized entropy confidence | 91.8% |
| Chosen-option probability ECE (10 bins per source) | 33.5% |

These are equal-weight averages of per-source results, **not** an accuracy pooled across all questions. The local validation used up to 32 rows per source, yielding 832 scored questions. The checkpoint beat the random expectation on 19/21 sources and the pilot on 17/21, but it **regressed by 2.1 percentage points** from stage 15. The random line is an analytical expectation, not a sampled model run. Entropy confidence measures how concentrated the option distribution is; it is not a probability that the decision is correct. ECE uses the selected option's actual probability.

### Where it struggles

| Validation source | Accuracy | Random expectation | Entropy confidence |
|:--|--:|--:|--:|
| Yelp Review Full | 6.3% | 20.0% | 100.0% |
| Amazon Reviews · EN | 18.8% | 20.0% | 100.0% |
| ARC-Challenge | 34.4% | 25.0% | 92.5% |
| BoolQ | 71.9% | 50.0% | 82.7% |
| AG News | 90.6% | 25.0% | 100.0% |

Yelp and Amazon are below random expectation in this small validation sample. The model was trained on all 15 smaller curriculum splits, then stopped partway through Amazon Reviews after 136,392 rows of that split. CodeReviewer, MultiNLI, DBpedia 14, Yelp, and Consumer Finance had **not** yet been training stages, although all were included in every cross-source validation. The very high confidence and poor review accuracy make the limitations concrete.

## What is in this repository

| File | Purpose |
|:--|:--|
| `model.safetensors` | BF16 fine-tuned encoder and FP32 decision head, with no optimizer or RNG state |
| `decision_config.json` | Architecture, base-model revision, checkpoint step, and SHA-256 provenance |
| `calibration.json` | Fitted temperature and held-out calibration metrics |
| `tokenizer/` | Tokenizer files pinned to the ModernBERT-large base revision |
| `assets/overview.png` | The visual score summary above |

The weights are provided in Safetensors so they can be inspected and loaded without unpickling the original training checkpoint.

## Load and score a request

Install the project and its pinned dependencies, then run this example from the project directory:

```bash
git clone https://github.com/Flydexo/decisions.git
cd decisions
uv sync
```

```python
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from decisions.evaluation import preprocessing
from decisions.model import inference_logits, load_model
from decisions.schema import prepare_request, to_device

repo = "flydexo/bad-laya"
config = json.load(open(hf_hub_download(repo, "decision_config.json")))
temperature = json.load(open(hf_hub_download(repo, "calibration.json")))["temperature"]
device = torch.device("cpu")
model, tokenizer = load_model(config["model"], config["ablation"], device)
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=True)
model.precision_settings["mixed_precision"] = "fp32"  # practical CPU inference
model.eval()

request = {
    "state": "The film was funny and moving.",
    "questions": {
        "sentiment": {
            "type": "choice",
            "instructions": "Choose the overall sentiment.",
            "criteria": {"negative": "negative review", "positive": "positive review"},
        }
    },
}
inputs = prepare_request(request, tokenizer, **preprocessing(config["model"], False))
with torch.no_grad():
    logits = inference_logits(model, to_device(inputs, device), question_microbatch_size=4)
probs = (logits.float() / temperature).softmax(-1)[0, : len(inputs["option_labels"][0])]
print(dict(zip(inputs["option_labels"][0], probs.tolist())))
```

The project tokenizer and base-model config are pinned to revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`. The convenience loader above downloads the base ModernBERT weights before replacing them with this checkpoint; plan disk space accordingly. The included `tokenizer/` files also allow local tokenizer loading.

## Temperature calibration

The published `calibration.json` holds one positive temperature fitted to raw logits from 832 validation questions by minimizing negative log-likelihood. Evaluation questions came from separate dataset partitions and were not used to choose the temperature. The validation sample had previously been used to select and inspect this checkpoint, so the separate evaluation result is the meaningful calibration check. Apply the temperature to **every valid option logit before softmax**. The selected answer and its accuracy do not change.

| Separate evaluation partitions · 832 questions | Original | Temperature 12.60 |
|:--|--:|--:|
| Accuracy | 65.4% | 65.4% |
| Mean chosen-answer probability | 91.2% | 60.1% |
| Negative log-likelihood | 6.08 | 1.13 |
| Chosen-answer probability ECE · 10 bins | 26.2% | 15.3% |

The original 91.8% figure above is normalized entropy confidence from the validation sample. For a decision forecast, use the probability of the chosen option after temperature scaling. A shared temperature can still be wrong for a particular source or a new domain. See the [calibration report](https://github.com/Flydexo/decisions/blob/main/reports/bad-laya-calibration.json) and [results page](https://github.com/Flydexo/decisions/blob/main/reports/curriculum-results.html) for the separate evaluation metrics and reliability plot.

## Training and evaluation provenance

- **Architecture:** `answerdotai/ModernBERT-large` encoder, two-layer transformer decision head, option-marker scoring; 1,024-token decision context.
- **Objective:** sampled reward using log, spherical, and ordinal ranked-probability components.
- **Training:** complete splits in ascending-size curriculum order; 15 stages completed, stage 16 (Amazon Reviews) interrupted; 398,779 cumulative training rows and 60,928 optimizer steps.
- **Checkpoint:** BF16 encoder and FP32 head exported from the locally evaluated `last_bf16_fresh_optimizer.pt`. `decision_config.json` records SHA-256 hashes of both the source checkpoint and published weights.
- **Validation:** 21 sources, seed 42, up to 32 rows per source, 832 scored questions; local FP32 CPU operations on the BF16 encoder weights.
- **Comparators:** uniform chance computed from valid option counts; bounded mixed-pool RTX 4090 pilot at step 672; complete stage-15 curriculum checkpoint.

The curriculum comparison above is **validation**, not a held-out final benchmark. The calibration check uses separate evaluation partitions but is also small. Small per-source samples, nonuniform option counts, and the difference in training budgets limit what the comparisons establish. Accuracy and confidence can change substantially across stages: the partial stage-16 checkpoint is weaker overall than stage 15. Sensitive domains represented among the sources include finance, customer support, phishing, and safety; do not use this model to make consequential decisions without a separate domain-specific evaluation and human oversight.

## Limitations and use

- **Useful experiments:** English text classification, routing, triage, and typed decision research where the allowed answers are supplied with each question. Refit or verify calibration on a labelled sample of the actual workload before choosing confidence thresholds.
- **Poor fit:** Open-ended generation, fact retrieval, image or video inputs, and fully automated decisions with medical, legal, financial, employment, or safety consequences.
- **Known failure modes:** Review sentiment collapsed in the partial stage-16 checkpoint: Yelp Review Full scored 6.3% and Amazon Reviews 18.8% on their small validation samples. Temperature reduces reported certainty but does not fix these answers. A 1,024-token context can also refuse or truncate long requests, depending on preprocessing settings.
- **Evaluation limits:** The 21-source validation set influenced checkpoint selection. The separate calibration evaluation has 832 questions and is too small to establish reliability within every source or on shifted domains. The 0.3 Decision Index run is pending; do not compare these local percentages to its leaderboard.

For the per-source results and training trajectory, see the [results page source](https://github.com/Flydexo/decisions/blob/main/reports/curriculum-results.html) and its [data snapshot](https://github.com/Flydexo/decisions/blob/main/reports/curriculum-results-data.json). The base encoder is [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) (Apache 2.0). A license for this combined checkpoint is not declared here.
