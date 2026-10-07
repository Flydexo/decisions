"""Fine-tune the whole encoder once, comparing to saved frozen pilots at equal exposure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from decisions.checkpoints import read_checkpoint
from decisions.data import dataset_configs, mixed_examples, prepare_training_cache
from decisions.trainer import train
from scripts.run_collapse_pilots import json_lines


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return hashlib.sha256(json.dumps({'state': row['state'], 'questions': row['questions']},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def at_step(rows, step):
    return next(row for row in rows if row['step'] == step)


def comparison(name, validation, probe):
    return {'name': name, 'step': validation['step'], 'rows': validation['rows'],
            'macro_accuracy': validation['macro_accuracy'], 'datasets': validation['datasets'],
            'encoder_marker_rms': probe['features/encoder/marker_rms'],
            'encoder_marker_cosine': probe['features/encoder/marker_cosine'],
            'head_marker_rms': probe['features/head/marker_rms'],
            'head_marker_cosine': probe['features/head/marker_cosine'],
            'dispersion_retention': probe['features/dispersion_retention'],
            'probe_rlcd_reward': probe['probe/rlcd_reward']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, default=ROOT / 'outputs/collapse_rlcd_pilot')
    parser.add_argument('--name', default='finetune_prenorm_rlcd_1024_smoke')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.pilot_root.resolve()
    directory = root / args.name
    if Path(args.name).name != args.name:
        parser.error('name must be a single directory name')
    if directory.exists() and not args.resume:
        parser.error('Existing run will not be overwritten; choose --name or --resume')
    if args.resume and not (directory / 'last.pt').is_file():
        parser.error('Resume needs an existing last.pt')
    old = json.loads((ROOT / 'reports/collapse_rlcd_pilot_summary.json').read_text())
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=['experiment=finetune_smoke_prenorm_rlcd', 'device=mps'])
    cfg.data.train_cache_dir = str(root / 'training_samples')
    if args.resume:
        cfg.resume = str(directory / 'last.pt')
    config = OmegaConf.to_container(cfg, resolve=True)
    datasets = dataset_configs(config)
    status_path = root / f'{args.name}_status.json'
    def status(phase, **values):
        status_path.write_text(json.dumps({'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                          'run_dir': str(directory), **values}, indent=2))
    try:
        status('checking_matched_samples')
        # This comparison requires the existing bounded HF-streamed samples;
        # never silently replace them with a new draw or download full datasets.
        hashes = {d['name']: digest(root / 'training_samples' / f"{d['name']}.jsonl") for d in datasets}
        if hashes != old['training_pool_sha256']:
            raise ValueError('Training pool hashes differ from the saved frozen pilots')
        if datasets != old['runs']['prenorm_rlcd']['config']['dataset_configs']:
            raise ValueError('Dataset schema/partition provenance differs from the frozen pilot')
        if old['runs']['prenorm_rlcd']['validation_sample_sha256'] != old['runs']['no_transformer_rlcd']['validation_sample_sha256']:
            raise ValueError('Frozen baselines used different validation samples')
        counts = prepare_training_cache(config, datasets, root / 'training_samples')
        directory.mkdir(parents=True, exist_ok=True)
        validation_dir = directory / 'validation_samples'
        validation_dir.mkdir(exist_ok=True)
        old_validation = root / 'prenorm_rlcd/validation_samples'
        overlap, validation_hashes = {}, {}
        for dataset in datasets:
            key = dataset['name']
            source = next(old_validation.glob(f'{key}-*.json'))
            if digest(source) != old['runs']['prenorm_rlcd']['validation_sample_sha256'][key]:
                raise ValueError(f'{key}: old validation sample hash changed')
            target = validation_dir / source.name
            if target.exists() and digest(target) != digest(source):
                raise ValueError(f'{key}: existing validation sample differs')
            if not target.exists():
                shutil.copyfile(source, target)
            rows = json.loads(target.read_text())['rows']
            training_ids = {identity(row) for row in json_lines(root / 'training_samples' / f'{key}.jsonl')}
            overlap[key] = len(training_ids & {identity(row) for row in rows})
            validation_hashes[key] = digest(target)
        if any(overlap.values()):
            raise ValueError('Training/validation input overlap')
        row_order = list(mixed_examples(datasets, 'train', seed=42, epoch=0,
                                        shuffle_buffer=2048, cache_dir=str(root / 'training_samples')))
        order_hash = hashlib.sha256(json.dumps(row_order, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        status('training', training_pool_counts=counts, training_validation_overlap=overlap)
        train(cfg, directory)
        status('summarizing')
        saved = read_checkpoint(directory / 'last.pt')
        progress, resolved = saved['progress'], saved['config']
        if progress['step'] != 64 or progress['rows'] != 256:
            raise RuntimeError('Smoke budget stopped before matched 64 updates / 256 rows; resume required')
        if not resolved['model']['train_encoder'] or resolved['model']['encoder_last_n_layers'] is not None:
            raise RuntimeError('Smoke must unfreeze the complete encoder')
        trainable = sum(t.numel() for section in ('head', 'encoder') for t in saved[section].values())
        # Sample several early/late trainable tensors against pinned pretrained
        # weights using safetensor slices, avoiding another full encoder in RAM.
        from huggingface_hub import try_to_load_from_cache
        from safetensors import safe_open
        initial_path = try_to_load_from_cache(resolved['model']['encoder'], 'model.safetensors',
                                              revision=resolved['model']['revision'])
        changes = {}
        if not isinstance(initial_path, str):
            raise RuntimeError('Pinned original encoder weights unavailable for the update audit')
        with safe_open(initial_path, framework='pt', device='cpu') as initial:
            for key in ['layers.0.attn.Wqkv.weight', 'layers.27.mlp.Wi.weight', 'final_norm.weight']:
                sl = (slice(0, 16), slice(0, 16)) if saved['encoder'][key].ndim == 2 else slice(0, 16)
                trained = saved['encoder'][key][sl].clone()
                original = initial.get_slice('model.' + key)[sl]
                changes[key] = (trained - original).abs().max().item()
        if not all(v > 0 for v in changes.values()):
            raise RuntimeError('Early/late encoder parameter update audit failed')
        del saved
        probes = json_lines(directory / 'feature_diagnostics.jsonl')
        history = json_lines(directory / 'validation_history.jsonl')
        metrics = json_lines(directory / 'metrics.jsonl')
        probe = at_step(probes, 64)
        validation = at_step(history, 64)
        matched = {name: comparison('Frozen · '+name, at_step(run['validation_history'], 64),
                                    at_step(run['diagnostics'], 64)) for name, run in old['runs'].items()}
        matched['finetuned_prenorm_rlcd'] = comparison('Whole encoder · pre-norm RLCD', validation, probe)
        initial_probe = probes[0]
        encoder_retention = probe['features/encoder/marker_rms'] / initial_probe['features/encoder/marker_rms']
        summary = {
            'protocol': 'One full-encoder 1024-token FP16 pre-norm RLCD smoke; saved frozen baselines compared at 64 updates / 256 row presentations. No new no-transformer run, final evaluation or benchmark samples.',
            'completed_at': datetime.now(timezone.utc).isoformat(), 'seed': 42, 'config': resolved,
            'progress': progress, 'trainable_parameters': trainable, 'encoder_parameter_sample_max_changes': changes,
            'training_pool_counts': counts, 'training_pool_sha256': hashes, 'training_row_order_sha256': order_hash,
            'validation_sample_sha256': validation_hashes, 'training_validation_input_overlap': overlap,
            'comparisons': matched, 'diagnostics': probes, 'validation_history': history,
            'selected_validation': at_step(history, progress['best_step']), 'metrics': metrics,
            'encoder_dispersion_retention_from_initial': encoder_retention,
            'collapse_on_fixed_probe': probe['features/dispersion_retention'] < .01 and probe['features/head/marker_cosine'] > .999,
            'encoder_collapse_on_fixed_probe': encoder_retention < .01 and probe['features/encoder/marker_cosine'] > .999,
            'peak_logged_driver_gib': max(m['memory/driver_gib'] for m in metrics),
            'post_update_live_memory_range_gib': [min(m['memory/live_gib'] for m in metrics), max(m['memory/live_gib'] for m in metrics)],
            'peak_process_rss_gib': max(m['memory/process_peak_rss_gib'] for m in metrics),
            'skipped_optimizer_updates': sum(not m['train/optimizer_updated'] for m in metrics),
            'checkpoint_sizes_gib': {p.name: p.stat().st_size / 2**30 for p in directory.glob('*.pt')},
            'trackio': {'project': resolved['logging']['project'], 'directory': str(root / 'trackio'), 'run': args.name},
            'limitations': ['One seed; 256 unique training rows and 128 reused validation rows across four sources.',
                            'FP16, microbatching and encoder learning rate differ from the original frozen FP32 pilots; this is not an isolated unfreezing ablation.',
                            'A short stability/learning smoke does not establish convergence or generalization.',
                            'RMS collapse rules are diagnostics, not guarantees. Encoder and head collapse are checked separately.',
                            'Logged memory snapshots are not a measurement of total physical RAM or internal kernel peaks.']}
        report = ROOT / 'reports/finetune_1024_smoke_summary.json'
        text = json.dumps(summary, indent=2, allow_nan=False)
        report.write_text(text+'\n')
        (directory / 'summary.json').write_text(text+'\n')
        status('completed', report=str(report), comparisons={k: v['macro_accuracy'] for k,v in matched.items()})
        print('Smoke completed: '+str(report), flush=True)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
