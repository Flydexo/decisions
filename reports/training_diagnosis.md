# Why the overnight model performed poorly

The strongest measured failure is option-representation collapse in the trainable
head. This diagnostic uses the validation-selected checkpoint and the same two
AG News training-pool examples as the earlier inference diagnostic. No optimizer
updates or checkpoint reselection were performed.

| Stage | Mean cosine similarity between option markers | RMS difference from the option mean |
| --- | ---: | ---: |
| Frozen encoder | 0.930015 | 0.268214 |
| After question-type embedding | 0.960179 | 0.268214 |
| After first trained transformer layer | 0.999771 | 0.013096 |
| After second trained transformer layer | 1.000000 | 0.0000933 |

The option differences shrink by approximately 2,875 times. A shared scorer
applied to almost identical vectors produces almost identical logits, hence
near-uniform probabilities. The four probabilities are approximately 0.25,
with only 0.000023–0.000025 separating their minimum and maximum.

A fresh randomly initialized reference head retains an RMS difference of
0.150967 after its second layer. It is a new reference, not the original run's
saved initialization. Removing the question-type embedding only at inference
increases the trained head's final RMS difference to 0.0003215: its common
offset contributes to similarity, but this intervention does not restore
discriminative features. A retrained ablation would be needed to assess it.

Across five training-mode head passes with fixed encoder features, dropout
produces per-option probability standard deviations of 0.009–0.027. This is far
larger than the surviving evaluation-time option signal. This supports checking
dropout and train/eval behavior; it does not prove dropout caused the collapse.

The earlier CPU/MPS comparison agreed within 7.45e-8 on these examples. This
small check provides no evidence of a GPU-specific numerical failure.

Other plausible contributors, not established causes:

- The 26.25-million-parameter head received only about 706 rows per dataset at
  the selected checkpoint, and at most about 1,221 per dataset in the final
  training state. Seventeen different tasks dilute a small supervision budget.
- Every option is represented by the same MASK token, with meaning supplied
  through its surrounding context. The frozen encoder's marker vectors already
  have high similarity. Pooling option-description tokens or scoring each
  input/option pair may provide a more robust signal.
- The code uses the default TransformerEncoder stack initialization. PyTorch
  documents that cloned layers start with identical parameter values and
  recommends initializing layers independently. This is a concrete setup
  concern, but it has not been isolated as the cause here. See the
  [official TransformerEncoder documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.transformer.TransformerEncoder.html).

Dataset audit problems explain misleading scores rather than this global
probability collapse: the FlakeFlagger final sample has no flaky cases, and
duplicate complaint narratives crossed the old ID-based holdout. These issues
remain documented in `all_large_audit.json`.

The first next experiment should be a small training-only overfit check with
the existing no-transformer head. Then compare matched validation runs with
and without the transformer, small or zero question-type offsets, independently
initialized layers, and reduced dropout. Preserve the original encoder features
through a residual path if retaining the additional transformer. Scale up only
after a simple head learns the training examples and improves clean validation.

Exact stage measurements are in `head_collapse_diagnostic.json`. The diagnostic
is small; broader training-only measurements are needed to establish the full
mechanism and verify a remedy.
