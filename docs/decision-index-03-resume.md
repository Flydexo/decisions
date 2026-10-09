# Resume bad-laya's Decision Index 0.3 run

Paused on 2026-10-09 at the user's request. The MPS inference process, parent
runner, and `caffeinate` helper were confirmed stopped. The 30-minute heartbeat
automation `complete-bad-laya-decision-index-0-3` is paused too. Do not resume
either until the user asks.

## Saved state

- Run: `outputs/decision_index_03/runs/bad-laya-full-03/`
- Canonical, hash-verified suite: `outputs/decision_index_03/suite-0.3/`
- Runner: `scripts/run_decision_index_03_sequential.py`
- Checkpoint and calibrated temperature are pinned in the runner. Use MPS, not CPU.
- 12 of 43 benchmarks finished (all 12 count toward the public index). ACOS is
  partial at 384/1,565 requests. The remaining 30 have not started. The
  `results.jsonl` file has 6,801 valid results.
- Snapshot score: `outputs/decision_index_03/runs/bad-laya-paused-03/scores.json`
  has `complete: false`. Its `decision_index` is **not a final score**.
- Aggregate table and pinned comparator snapshot:
  `reports/bad-laya-decision-index-03-paused.json`. ARC-Easy is a separate,
  completed run in `outputs/decision_index_03/runs/bad-laya-arc-easy/`.
- `sequential-progress.json` still names ACOS as active because the process was
  interrupted. Treat it as a saved stage pointer, not proof of a live process.

## Before resuming

1. Confirm the user has asked to continue. Check disk space; the runner stops
   before a stage if less than 4 GiB is free.
2. Check `ps` for the runner and its `decision_index run` child. The stale PID
   in `sequential.pid` is not sufficient evidence that either is alive. Do not
   start a second worker if one exists.
3. Check `sequential.log`, `38.log`, `sequential-progress.json`, and the last
   complete line of `results.jsonl`. Preserve `test.py`, which is unrelated.
4. Confirm `.venv/bin/python` reports `torch.backends.mps.is_available()`.

```sh
cd /Users/matheoledevehat/Code/decisions
df -h .
ps -p "$(cat outputs/decision_index_03/runs/bad-laya-full-03/sequential.pid)" -o pid,etime,command
ps -axo pid,ppid,etime,command | rg 'decision_index run|run_decision_index_03_sequential.py'
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'
```

## Resume one benchmark at a time

Run this in a shell with Apple GPU access. The runner takes a file lock,
rebuilds its per-benchmark request files from the **canonical** suite, skips
completed `run_id`s, and resumes ACOS before moving to the next dataset. It
checks stage completeness and stops on errors. `caffeinate` keeps the Mac awake.

```sh
cd /Users/matheoledevehat/Code/decisions
caffeinate -i .venv/bin/python -u scripts/run_decision_index_03_sequential.py \
  >> outputs/decision_index_03/runs/bad-laya-full-03/sequential.log 2>&1
```

Do not run inference on CPU or rent a Vast GPU. If a stage fails, inspect its
numbered log and `results.jsonl`, then rerun the same command after fixing the
cause. The runner's file lock prevents accidental concurrent starts.

## Finish and publish

After the runner completes, score the saved results against the verified suite:

```sh
.venv/bin/python -m decision_index score --edition 0.3 \
  --suite-dir outputs/decision_index_03/suite-0.3 \
  --results outputs/decision_index_03/runs/bad-laya-full-03/results.jsonl \
  --engine decisions.benchmark:DecisionEngine \
  --out outputs/decision_index_03/runs/bad-laya-full-03
```

Require `scores.json` to say `complete: true`, account for all 43 entries, and
verify the result payload hashes against the canonical suite before calling
the public index complete. The board's **Full score** includes private tests
that only its maintainers run. Regenerate aggregate comparisons with
`scripts/build_decision_index_03_partial_report.py` (adapt its score path and
completion assertion for the final run), update the README and model card,
then upload verified results using the official kit. Do not upload the current
partial snapshot as an official leaderboard result. Push repository edits to
Git before finishing.
