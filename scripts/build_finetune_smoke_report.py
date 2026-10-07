"""Build the smoke recap from recorded measurements, without running evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'reports/finetune_1024_smoke_summary.json').read_text())
    comparisons = report['comparisons']
    lines = ['# Full-encoder ModernBERT-large smoke at 1,024 tokens', '', report['protocol'], '',
             '## Matched comparison', '',
             'All three rows below are measured at **64 optimizer-update attempts and 256 training-row presentations**. The frozen baselines are saved observations from the previous pilots; only the whole-encoder pre-norm variant was trained in this smoke.', '',
             '| Variant | Mean validation accuracy | Encoder option RMS | Head option RMS | Encoder cosine | Head cosine | Probe RLCD reward |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in comparisons.values():
        lines.append(f"| {row['name']} | {row['macro_accuracy']:.1%} | {row['encoder_marker_rms']:.4f} | {row['head_marker_rms']:.4f} | {row['encoder_marker_cosine']:.4f} | {row['head_marker_cosine']:.4f} | {row['probe_rlcd_reward']:.4f} |")
    full = comparisons['finetuned_prenorm_rlcd']
    frozen = comparisons['prenorm_rlcd']
    mean_nll = lambda row: sum(d['nll'] for d in row['datasets'].values()) / len(row['datasets'])
    improved = [name for name, score in full['datasets'].items() if score['accuracy'] > frozen['datasets'][name]['accuracy']]
    improved_text = ', '.join(improved) if improved else 'none'
    lines += ['', f"Datasets with higher validation accuracy than frozen pre-norm: {improved_text}. Calibration worsens: mean validation NLL is **{mean_nll(full):.2f}**, versus **{mean_nll(frozen):.2f}** for frozen pre-norm. BoolQ entropy confidence is {full['datasets']['boolq']['entropy_confidence']:.3%} at {full['datasets']['boolq']['accuracy']:.1%} accuracy, matching the always-true baseline for this 19-true/13-false sample. Preventing feature collapse does not establish reliable decisions."]
    lines += ['', '| Dataset | Frozen pre-norm | Frozen no transformer | Whole encoder + pre-norm |',
              '| --- | ---: | ---: | ---: |']
    for name in report['training_pool_counts']:
        scores = [f"{r['datasets'][name]['accuracy']:.1%}" for r in comparisons.values()]
        lines.append('| '+name+' | '+' | '.join(scores)+' |')
    lines += ['', '## Calibration on the same validation rows', '',
              '| Variant / dataset | NLL | Brier | Mean entropy confidence | Probability ECE |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in comparisons.values():
        for name, score in row['datasets'].items():
            lines.append(f"| {row['name']} / {name} | {score['nll']:.4f} | {score['brier']:.4f} | {score['entropy_confidence']:.1%} | {score['probability_ece']:.4f} |")
    first, last = report['diagnostics'][0], report['diagnostics'][-1]
    updated = report['progress']['step'] - report['skipped_optimizer_updates']
    lines += ['', '## Stability and actual encoder updates', '',
              f"- All {report['trainable_parameters']:,} encoder/head parameters trainable; FP16 forward operations with FP32 parameter, gradient and Adam storage.",
              '- Full RLCD: log + 0.5 × spherical − ordinal RPS, with 32 candidates and sigma 1. No CE-only substitution.',
              '- One-row microbatches, four-row accumulation, non-reentrant encoder/head checkpointing; encoder LR 1e-5 and head LR 1e-4.',
              f"- {updated} successful updates, {report['skipped_optimizer_updates']} skipped by gradient scaling.",
              f"- Encoder option RMS on eight fixed training examples: {first['features/encoder/marker_rms']:.4f} → {last['features/encoder/marker_rms']:.4f} ({report['encoder_dispersion_retention_from_initial']:.3f} × initial).",
              f"- Head/encoder RMS ratio at the end: {last['features/dispersion_retention']:.4f}; head cosine: {last['features/head/marker_cosine']:.4f}.",
              f"- Head-collapse rule flagged: {report['collapse_on_fixed_probe']}; encoder-collapse rule flagged: {report['encoder_collapse_on_fixed_probe']}.",
              f"- Probe reward: {first['probe/rlcd_reward']:.4f} → {last['probe/rlcd_reward']:.4f}. Stochastic RLCD gradient-estimator loss is not CE and does not rank model quality.", '',
              'The head-collapse rule checks a head/encoder RMS ratio below 0.01 and head cosine above 0.999. The encoder rule checks encoder RMS below 0.01 × its initial value and encoder cosine above 0.999. These are diagnostics on the fixed probe, not general guarantees.', '',
              'Early and late sampled parameter changes against the pinned pretrained encoder:', '']
    for key, value in report['encoder_parameter_sample_max_changes'].items():
        lines.append(f'- `{key}`: max absolute change {value:.8g}.')
    low, high = report['post_update_live_memory_range_gib']
    mean = sum(m['train/seconds'] for m in report['metrics']) / len(report['metrics'])
    lines += ['', '## Resources and logging', '',
              f"- MPS allocation cap: 9 GiB. Peak logged driver allocation: {report['peak_logged_driver_gib']:.2f} GiB.",
              f'- Post-update live allocations: {low:.3f}–{high:.3f} GiB. Peak process RSS: '+f"{report['peak_process_rss_gib']:.2f} GiB.",
              f'- Mean update time: {mean:.2f} seconds (periodic validation and checkpoint serialization excluded).',
              '- These separate counters do not measure total physical RAM or internal kernel peaks.',
              f"- Trackio project `{report['trackio']['project']}`, run `{report['trackio']['run']}`.",
              f"- Checkpoints: "+', '.join(f'{k} {v:.2f} GiB' for k,v in report['checkpoint_sizes_gib'].items())+'.', '',
              '## Sampling and limitations', '',
              'The same bounded HF-streamed training pools and 32 validation rows per source were reused and checked by SHA-256. Training/validation input overlaps are zero. The first epoch follows the same deterministic row order as the frozen pilots. No final test split or Decision Index request was read.', '']
    lines.extend('- '+item for item in report['limitations'])
    lines += ['', 'The longer frozen pilots ran 256 updates / 1,024 presentations; their terminal scores are not the matched comparison above. Validation checkpoint selection is separate from final evaluation, which was not run.', '',
              '```sh', '.venv/bin/python scripts/run_finetune_smoke.py', '# Exact recovery after confirming no matching GPU worker is alive:',
              '.venv/bin/python scripts/run_finetune_smoke.py --resume', '.venv/bin/python scripts/show_pilot_dashboard.py --check', '```', '',
              '[Interactive recap](results.html) · [All aggregate measurements](finetune_1024_smoke_summary.json)', '']
    (ROOT / 'reports/finetune_1024_smoke.md').write_text('\n'.join(lines))
    print('Built reports/finetune_1024_smoke.md')


if __name__ == '__main__':
    main()
