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

**Explore:** [Decision Index 0.3 results](#decision-index-03-benchmarks) · [Quickstart](#load-and-score-a-request) · [Calibration](#temperature-calibration)

> **Research checkpoint.** Results vary by task, and the 0.3 run is paused before completion. The table below reports every completed benchmark and the partial ACOS stage; no overall public-index score is available.

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

## Decision Index 0.3 benchmarks

The [verified public 0.3 suite](https://github.com/Flydexo/decisions/blob/main/reports/decision-index-03-canonical-suite.json) was run one benchmark at a time on Apple MPS. The run paused after 12 of 43 benchmarks; ARC-Easy was completed in a separate run. **These are partial public results, not a full index or leaderboard submission.**

| Benchmark / metric | bad-laya | Answered | Chance / baseline | Laya | Kev 0.8B | Kev 4B | Cloudflare clef |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ToolRet · nDCG@10 | 13.3% | 548/685 | 13.4% | 12.7% | 51.7% | 61.0% | 69.1% |
| API-Bank · accuracy | — | 0/508 | 1.9% | 11.4% | 43.7% | 53.3% | 91.9% |
| Home appliance · case exact accuracy | — | 0/88 | 0.0% | 0.0% | 0.0% | 14.8% | 80.7% |
| ContractNLI · macro-F1 | 1.7% | 14/123 | 30.9% | 28.7% | 40.9% | 64.4% | 81.3% |
| GPQA Diamond · accuracy | 26.0% | 195/196 | 25.0% | 27.6% | 33.2% | 37.8% | 48.5% |
| ARC-Easy (shown only) · accuracy | 53.7% | 2376/2376 | 25.0% | 47.0% | 82.2% | 97.2% | 99.0% |
| WinoGrande · accuracy | 50.9% | 1267/1267 | 50.0% | 50.5% | 52.8% | 70.2% | 93.5% |
| MuSR · accuracy | 14.5% | 383/752 | 37.1% | 43.2% | 48.9% | 56.2% | 83.8% |
| BRIGHT · nDCG@10 | 4.2% | 78/220 | 11.6% | 19.9% | 31.8% | 39.0% | 47.5% |
| ACOS (partial) · per-review F1 | 0.15% | 384/1565 | 3.1% | 3.5% | 9.9% | 9.8% | 33.2% |
| FinEntity · macro-F1 | 52.0% | 979/979 | 32.0% | 61.0% | 71.4% | 87.1% | 96.1% |
| CRUXEval · accuracy | 38.8% | 570/570 | 37.0% | 40.2% | 37.7% | 48.1% | 86.5% |
| HLE · accuracy | 13.2% | 471/501 | 16.4% | 14.0% | 13.8% | 10.2% | 12.8% |
| New Yorker captions · accuracy | 26.1% | 528/528 | 20.0% | 27.1% | 27.8% | 52.3% | 70.3% |

Scores are the kit’s **coverage-adjusted raw metric**, so unsupported requests count against bad-laya. The “Chance / baseline” column is the board’s task-specific random/reference baseline; accuracy, F1 and nDCG are distinct metrics. A dash means the model answered no requests. **ACOS is partial**: its 0.15% interim figure covers only 384 of 1,565 requests and must not be treated as a full-benchmark comparison. ARC-Easy comes from a separate completed run and is shown on the board but does not count toward the public index. The other 12 completed rows are from the paused sequential run. Peer values come from the [public Decision Index snapshot](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/e452ca53f88e735031ca0605559c7d83fd1aa1b6/data/index.json) (generated 2026-10-07); each peer result has the same request count for the row. The full 0.3 public index is unavailable while 31 benchmarks remain. The board’s Full score also requires private tests.

The [aggregate report](https://github.com/Flydexo/decisions/blob/main/reports/bad-laya-decision-index-03-paused.json) gives exact scores and provenance. WinoGrande scored 50.9% in this sequential run; an earlier isolated run scored 50.7%.

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

For a decision forecast, use the probability of the chosen option after temperature scaling; normalized entropy confidence describes distribution sharpness instead. A shared temperature can still be wrong for a particular source or a new domain. See the [calibration report](https://github.com/Flydexo/decisions/blob/main/reports/bad-laya-calibration.json) and [results page](https://github.com/Flydexo/decisions/blob/main/reports/curriculum-results.html) for the separate evaluation metrics and reliability plot.
