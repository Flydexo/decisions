"""Run two matched, bounded RLCD pilots and save aggregate collapse diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from decisions.checkpoints import read_checkpoint
from decisions.data import dataset_configs, prepare_training_cache
from decisions.trainer import clear_cache, train


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/collapse_rlcd_pilot')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--minutes-per-run', type=float, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.minutes_per_run <= 0:
        parser.error('steps and minutes-per-run must be positive')
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    variants = [('prenorm_rlcd', 'collapse_prenorm_rlcd'),
                ('no_transformer_rlcd', 'collapse_no_transformer_rlcd')]
    for name, _ in variants:
        if (root / name / 'last.pt').exists():
            raise ValueError(f'{root / name}: an existing pilot will not be overwritten; choose a new run-dir')
    status_path = root / 'status.json'

    def status(phase, **values):
        status_path.write_text(json.dumps(dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(),
                                              **values), indent=2))

    try:
        configs = []
        with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
            for name, experiment in variants:
                cfg = compose(config_name='config', overrides=[f'experiment={experiment}', 'device=mps'])
                cfg.training.max_steps = args.steps
                cfg.training.epochs = math.ceil(args.steps / 64)
                cfg.training.max_seconds = args.minutes_per_run * 60
                cfg.data.train_cache_dir = str(root / 'training_samples')
                configs.append((name, cfg))
        status('preparing_training_pools')
        first = OmegaConf.to_container(configs[0][1], resolve=True)
        datasets = dataset_configs(first)
        counts = prepare_training_cache(first, datasets, root / 'training_samples')
        runs = {}
        for name, cfg in configs:
            status('training', variant=name)
            directory = root / name
            train(cfg, directory)
            saved = read_checkpoint(directory / 'last.pt')
            progress, config = saved['progress'], saved['config']
            trainable = sum(value.numel() for value in saved['head'].values())
            assert config['ablation']['objective'] == 'sampled_reward'
            assert all(config['ablation']['reward_weights'][k] > 0 for k in ['log', 'spherical', 'rps'])
            del saved
            diagnostics = json_lines(directory / 'feature_diagnostics.jsonl')
            history = json_lines(directory / 'validation_history.jsonl')
            metrics = json_lines(directory / 'metrics.jsonl')
            selected = next(v for v in history if v['step'] == progress['best_step'])
            overlap, validation_hashes, validation_labels = {}, {}, {}
            identity = lambda row: hashlib.sha256(json.dumps(
                {'state':row['state'], 'questions':row['questions']}, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest()
            for dataset in datasets:
                key = dataset['name']
                training_ids = {identity(row) for row in json_lines(root / 'training_samples' / f'{key}.jsonl')}
                sample = next((directory / 'validation_samples').glob(f'{key}-*.json'))
                validation_rows = json.loads(sample.read_text())['rows']
                validation_ids = {identity(row) for row in validation_rows}
                validation_hashes[key] = hashlib.sha256(sample.read_bytes()).hexdigest()
                validation_labels[key] = dict(Counter(max(target, key=target.get) for row in validation_rows
                                                       for target in row['targets'].values()))
                overlap[key] = len(training_ids & validation_ids)
            assert not any(overlap.values()), 'Training/validation input overlap'
            last = diagnostics[-1]
            runs[name] = {
                'config':config, 'progress':progress, 'trainable_parameters':trainable,
                'diagnostics':diagnostics, 'validation_history':history,
                'selected_validation':selected, 'final_validation':history[-1],
                'training_validation_input_overlap':overlap,
                'validation_sample_sha256':validation_hashes,
                'validation_gold_counts':validation_labels,
                'collapse_on_fixed_probe':last['features/dispersion_retention'] < .01 and
                                           last['features/head/marker_cosine'] > .999,
                'live_memory_range_gib':[min(m['memory/live_gib'] for m in metrics),
                                         max(m['memory/live_gib'] for m in metrics)],
                'peak_driver_gib':max(m['memory/driver_gib'] for m in metrics),
                'peak_process_rss_gib':max(m['memory/process_peak_rss_gib'] for m in metrics),
                'trackio_run':name,
            }
            clear_cache(torch.device('mps'))
            print('Pilot finished: ' + json.dumps({'variant':name, 'progress':progress,
                  'final_probe':last, 'selected_validation_accuracy':selected['macro_accuracy']}), flush=True)
        assert runs['prenorm_rlcd']['validation_sample_sha256'] == runs['no_transformer_rlcd']['validation_sample_sha256']
        summary = {
            'protocol':'Matched small pilots; fresh frozen ModernBERT-large, identical training pools and validation partitions. No final evaluation or benchmark tuning.',
            'seed':42, 'training_pool_counts':counts,
            'training_gold_counts':{d['name']:dict(Counter(max(target,key=target.get)
                for row in json_lines(root / 'training_samples' / f"{d['name']}.jsonl")
                for target in row['targets'].values())) for d in datasets},
            'training_pool_sha256':{d['name']:hashlib.sha256((root / 'training_samples' /
                                      f"{d['name']}.jsonl").read_bytes()).hexdigest() for d in datasets},
            'collapse_rule':'Head/encoder marker RMS ratio below 0.01 AND mean head option-marker cosine above 0.999, on the fixed eight-row training probe. This is a diagnostic threshold, not a general guarantee.',
            'limitations':['Two joint architecture variants, not isolated pre-norm versus initialization ablations.',
                          'One seed, only 64 unique training rows and 32 validation rows per dataset.',
                          'Training-mode RLCD gradient-estimator loss is not CE or a direct learning-quality measure; probe rewards and validation scores are logged separately.',
                          'Preventing feature collapse does not establish benchmark performance.'],
            'trackio':{'project':'decisions-collapse-rlcd', 'directory':str(root / 'trackio'),
                       'runs':[name for name, _ in variants]},
            'runs':runs,
        }
        report = ROOT / 'reports/collapse_rlcd_pilot_summary.json'
        report.write_text(json.dumps(summary, indent=2))
        (root / 'summary.json').write_text(json.dumps(summary, indent=2))
        status('completed', report=str(report))
        print('Report: ' + str(report), flush=True)
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
