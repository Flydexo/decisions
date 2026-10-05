"""Bounded all-source training, validation selection, then untouched evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from decisions.checkpoints import read_checkpoint
from decisions.benchmark import source_pilot
from decisions.data import dataset_configs, prepare_training_cache
from decisions.evaluation import Predictor, evaluate
from decisions.trainer import clear_cache, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=2)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/all_large')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    if args.hours <= 0:
        raise ValueError('hours must be positive')
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    def status(phase, **details):
        (run_dir / 'status.json').write_text(json.dumps({'phase': phase,
            'updated_at': datetime.now(timezone.utc).isoformat(), 'hours': args.hours,
            'run_dir': str(run_dir), **details}, indent=2))
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'conf')):
        cfg = compose(config_name='config', overrides=['experiment=all_large', 'device=mps'])
    cfg.training.max_seconds = int(args.hours * 3600)
    cfg.data.train_cache_dir = str(run_dir / 'training_samples')
    if args.resume:
        cfg.resume = str(args.resume.resolve())
    if not args.eval_only:
        if (run_dir / 'last.pt').exists() and not args.resume:
            raise ValueError('Existing run found: use --resume or a different --run-dir')
        free = shutil.disk_usage(run_dir).free / 2**30
        if free < 1.2:
            raise RuntimeError(f'Only {free:.2f} GiB free: need at least 1.2 GiB for training pools, atomic checkpoints and reserve')
        config = OmegaConf.to_container(cfg, resolve=True)
        datasets = dataset_configs(config)
        status('preparing')
        prepare_training_cache(config, datasets, run_dir / 'training_samples')
        gc.collect()
        status('training')
        train(cfg, run_dir)
    checkpoint = run_dir / 'best.pt'
    if not checkpoint.exists():
        raise RuntimeError('No validation-selected checkpoint; final evaluation will not run')
    clear_cache(torch.device('mps'))
    saved = read_checkpoint(checkpoint)
    status('evaluating', selected_step=saved['progress']['step'])
    predictor = Predictor(checkpoint, device='mps')
    evaluation = dict(saved['config']['evaluation'], role='eval', every_steps=0,
                      max_rows=256, preprocessing=saved['config']['model'])
    print('Final evaluation: only the validation-selected checkpoint, never used for model selection.', flush=True)
    report = evaluate(predictor.model, predictor.tokenizer, saved['config']['dataset_configs'],
                      predictor.device, evaluation, saved['config']['seed'])
    summary = {'protocol':'validation-selected ModernBERT-large; untouched final evaluation',
               'model':saved['config']['model'], 'ablation':saved['config']['ablation'],
               'best_progress':saved['progress'], 'training_pool_rows_per_dataset':saved['config']['data']['train_pool_rows'],
               'final_evaluation':report, 'final_evaluation_max_rows_per_dataset':256,
               'checkpoint':str(checkpoint),
               'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (run_dir / 'final_evaluation.json').write_text(json.dumps(summary,indent=2))
    (ROOT / 'reports/all_large_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2), flush=True)
    del predictor, saved
    clear_cache(torch.device('mps'))
    status('benchmarking')
    benchmark = OmegaConf.to_container(OmegaConf.load(ROOT / 'conf/benchmark.yaml'), resolve=True)
    benchmark.update(checkpoint=str(checkpoint), device='mps', question_batch_size=1)
    benchmark_dir = run_dir / 'benchmark'
    benchmark_dir.mkdir(exist_ok=True)
    source_pilot(benchmark, benchmark_dir)
    status('completed', selected_step=summary['best_progress']['step'])
    from build_results_page import main as build_page
    build_page()


if __name__ == '__main__':
    main()
