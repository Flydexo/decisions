# RTX 4090 ordered full-split run

This run streams one complete training partition at a time, in increasing order
of published raw training size. It covers the 21 sources in the RTX 4090 pilot
mixture, excluding MS MARCO and CommitPackFT. Source filters and separate
validation holdouts reduce the number of rows actually trained. The plan records
the raw-size order; each completed stage records its actual row count.

After every dataset, the runner evaluates a fixed, cached validation sample of
up to 32 rows from **all 21 sources**, including sources it has not trained yet.
The final test/evaluation partitions are not used for these stage reports. It
writes `stages/NN-source.pt` (inference weights), `stages/NN-source.json` (full
validation report), and `last.pt` (model, optimizer, and exact resume state).
The latest cross-source results are also in `validation_history.jsonl` and
Trackio project `decisions-rtx4090-curriculum` on the configured server.

On the 4090 instance, with `TRACKIO_WRITE_TOKEN` set in the environment:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_curriculum.py --plan
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_curriculum.py --hours 8
```

The `--hours` limit is cumulative training wall time, including source loading,
evaluation, and checkpoints. To continue after the limit, increase it and resume
from the same directory:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_curriculum.py --resume --hours 24
```

Use a new `--run-dir` for a separate experiment. The resume loader checks the
dataset and evaluation provenance and restores the optimizer and random states.
It streams past completed rows in the current source before continuing updates;
the scan can take time for a large source. A full run is roughly 9.6 million
raw rows before filtering, so it requires substantially more GPU time than the
pilot. The 8-hour limit prevents a single unattended run from consuming the
remaining Vast balance.
