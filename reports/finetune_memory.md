# Unfrozen ModernBERT-large: context, mixed precision and checkpoints

Full unfreezing trains all 421,029,889 encoder/head parameters. The M1 partial preset trains 50,763,777 parameters in the last two encoder blocks, final normalization and full head. Weights, gradients and Adam moments remain FP32 (6.27 GiB for full training, 2.14 GiB for partial training before activations). BF16/FP16 autocast changes suitable encoder and head forward operations. RLCD loss, probabilities and normalized entropy use FP32.

| Tokens | Forward precision | Initial loss scale | Encoder blocks trained | Questions | Accumulation | Result | Observed MPS driver allocation | Peak process RSS | Seconds/update |
| ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 256 | fp32 | — | All 28 | 1 | 1 | ok | 7.09 GiB | 0.72 GiB | 3.1 |
| 512 | fp32 | — | All 28 | 1 | 1 | ok | 7.02 GiB | 0.79 GiB | 3.7 |
| 512 | fp32 | — | All 28 | 5 | 1 | ok | 7.06 GiB | 0.82 GiB | 13.4 |
| 1024 | fp32 | — | All 28 | 1 | 1 | ok | 8.02 GiB | 0.79 GiB | 6.5 |
| 1024 | fp32 | — | All 28 | 1 | 2 | ok | 8.05 GiB | 0.77 GiB | 11.0 |
| 1536 | fp32 | — | All 28 | 1 | 2 | ok | 8.02 GiB | 0.77 GiB | 16.7 |
| 1536 | fp32 | — | All 28 | 1 | 8 | ok | 8.02 GiB | 0.76 GiB | 60.7 |
| 2048 | bf16 | — | All 28 | 1 | 8 | failed | Failed; see JSON error | 0.77 GiB | — |
| 2048 | bf16 | — | All 28 | 1 | 8 | failed | Failed; see JSON error | 0.60 GiB | — |
| 2048 | fp16 | 65536.0 | All 28 | 1 | 1 | failed | Failed; see JSON error | 0.77 GiB | — |
| 2048 | fp16 | 1024 | All 28 | 1 | 1 | ok | 8.06 GiB | 0.78 GiB | 11.6 |
| 2048 | fp16 | 1024 | All 28 | 5 | 1 | failed | Failed; see JSON error | 0.78 GiB | — |
| 2048 | fp16 | 1024 | 2 | 5 | 2 | ok | 5.03 GiB | 0.99 GiB | 118.2 |
| 2048 | fp16 | 65536.0 | All 28 | 1 | 8 | failed | Failed; see JSON error | 0.78 GiB | — |
| 2048 | fp32 | — | All 28 | 1 | 1 | ok | 8.03 GiB | 0.69 GiB | 13.6 |
| 2048 | fp32 | — | All 28 | 1 | 2 | failed | Failed; see JSON error | 0.70 GiB | — |
| 3072 | fp32 | — | All 28 | 1 | 1 | ok | 8.71 GiB | 0.60 GiB | 22.0 |
| 3072 | fp32 | — | All 28 | 1 | 2 | failed | Failed; see JSON error | 0.77 GiB | — |
| 3584 | fp32 | — | All 28 | 1 | 1 | failed | Failed; see JSON error | 0.76 GiB | — |
| 4096 | fp32 | — | All 28 | 1 | 1 | failed | Failed; see JSON error | 0.73 GiB | — |

## Interpretation

FP32 controls: 3,072 tokens passed with one row per update; 3,584 and 4,096 ran out of memory. With gradients retained across microbatches, 1,536 passed (including eight-row accumulation), while 2,048 and 3,072 ran out of memory. Those controls do not establish an AMP limit.

The requested preset keeps 2,048 tokens, FP16 autocast with initial scale 1,024, one-row/one-question forwards, one row per optimizer update, encoder/head checkpointing, full RLCD, encoder LR 1e-5 and head LR 1e-4. See the AMP rows above for actual results. FP16 uses dynamic gradient scaling: unscale once after accumulation, clip, step, update the scaler and save its state for exact resume. BF16 does not require gradient scaling.

The full-encoder five-question FP16 test also exceeded the cap at 2,048 tokens. The separate `finetune_large_m1` preset trains the last two encoder blocks and final normalization, plus the full head, with two-row accumulation. It retains the requested 2,048-token context and joint RLCD objective; its five-question, two-row accumulation preflight passed two updates at 5.03 GiB observed MPS driver allocation, with finite gradients and encoder weight updates. AMP evaluation forwards were finite. Full checkpoint save/resume passed: last.pt 1.95 GiB, inference best.pt 1.57 GiB; temporary files were removed.

ModernBERT's architectural limit is 8,192 tokens. Context includes state, instructions, options and special tokens. BF16 and FP16 both ran out of memory with eight-row accumulation at 2,048 tokens. FP16 passed three one-row updates with a 1,024 initial loss scale, at 8.06 GiB observed driver allocation. The higher 65,536 initial scale caused nonfinite gradients in the one-row test. These are short synthetic memory checks, not convergence results or an exhaustive context maximum. Shapes, extra options and allocator reservation can change the limit. Driver allocation and process RSS are separate counters, not a measurement of total physical system RAM.

A five-question FP32 row passed at 512 tokens, including full MPS save/resume: last.pt was 4.71 GiB and inference-only best.pt 1.57 GiB. Temporary test checkpoints were removed. Encoder gradients and weight updates were verified. Earlier frozen configs/checkpoints remain supported. Training metrics, loss scale and skipped optimizer updates are logged to Trackio.

```sh
uv run python train.py experiment=finetune_large_m1 device=mps
# Reproduce AMP memory checks without reading any dataset/evaluation rows:
.venv/bin/python scripts/preflight_finetune.py --max-len 2048 --mixed-precision fp16 --trackio
```

Sources: [Laya MPS recipe](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py), [Transformers memory accounting](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy), [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html), [MPS BF16 support](https://github.com/pytorch/pytorch/issues/139386).
