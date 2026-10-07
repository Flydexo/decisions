import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from decisions.cuda_pilot import check_environment, configure_cuda_budget, synthetic_rows, validate_pilot_config
from decisions.schema import collate
from decisions.trainer import memory
from scripts.run_rtx4090_pilot import worker_lock
from test_pipeline import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def pilot_config():
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        return OmegaConf.to_container(compose(config_name='config', overrides=['experiment=rtx4090_pilot']), resolve=True)


class CudaPilotTests(unittest.TestCase):
    def test_plan_needs_no_gpu_or_dataset_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'absent'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_rtx4090_pilot.py'),
                                     '--plan', '--run-dir', str(path)], capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['source_count'], 21)
            self.assertEqual(plan['maximum_training_pool_rows'], 2688)
            self.assertFalse(path.exists())
            names = plan['config']['mixture']
            self.assertNotIn('ms_marco', names)
            self.assertNotIn('commitpackft', names)

    def test_smaller_row_microbatch_preserves_four_row_effective_batch(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_rtx4090_pilot.py'),
                                 '--plan', '--batch-size', '1', '--question-microbatch-size', '1'],
                                capture_output=True, text=True, check=True)
        settings = json.loads(result.stdout)['config']['training']
        self.assertEqual(settings['batch_size'] * settings['gradient_accumulation_steps'], 4)
        self.assertEqual(settings['question_microbatch_size'], 1)

    def test_protocol_rejects_frozen_encoder_missing_transformer_ce_and_final_selection(self):
        base = pilot_config()
        self.assertEqual(len(validate_pilot_config(base)), 21)
        for section, key, value in [('model', 'train_encoder', False), ('model', 'max_len', 2048),
                                    ('ablation', 'no_transformer', True), ('ablation', 'objective', 'cross_entropy'),
                                    ('training', 'validation_role', 'eval')]:
            with self.subTest(section=section, key=key):
                cfg = copy.deepcopy(base)
                cfg[section][key] = value
                with self.assertRaises(ValueError):
                    validate_pilot_config(cfg)

    def test_full_length_synthetic_rows_preserve_77_options_and_joint_questions(self):
        cfg = pilot_config()
        inputs, target, spans = collate(synthetic_rows(4), Tokenizer(), max_len=1024,
                                        head_max_len=384, option_max_len=64, preserve_options=True)
        self.assertEqual(tuple(inputs['input_ids'].shape), (20, 1024))
        self.assertTrue((inputs['attention_mask'].sum(1) == 1024).all())
        self.assertEqual(spans, [(i * 5, (i + 1) * 5, 77) for i in range(4)])
        self.assertEqual(inputs['marker_mask'][0].sum().item(), 77)
        torch.testing.assert_close(target.sum(1), torch.ones(20))
        self.assertEqual(cfg['training']['candidates'], 32)

    def test_cuda_unavailable_fails_instead_of_using_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU/MPS fallback'):
                check_environment(pilot_config(), ROOT)

    def test_cuda_cap_reserves_headroom_and_rejects_impossible_budget(self):
        device = torch.device('cuda:0')
        with patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=24 * 2**30)), \
             patch('torch.cuda.set_per_process_memory_fraction') as setter:
            configure_cuda_budget(21, device)
            setter.assert_called_once_with(21 / 24, device)
            for invalid in [0, -1, 25, True, '21', float('nan')]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    configure_cuda_budget(invalid, device)

    def test_cuda_metrics_record_internal_peak_not_only_post_update_memory(self):
        with patch('torch.cuda.memory_allocated', return_value=2 * 2**30), \
             patch('torch.cuda.memory_reserved', return_value=3 * 2**30), \
             patch('torch.cuda.max_memory_allocated', return_value=8 * 2**30), \
             patch('torch.cuda.max_memory_reserved', return_value=9 * 2**30):
            result = memory(torch.device('cuda'))
            self.assertEqual(result['memory/live_gib'], 2)
            self.assertEqual(result['memory/peak_allocated_gib'], 8)
            self.assertEqual(result['memory/peak_reserved_gib'], 9)

    def test_worker_lock_rejects_duplicate_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'forced'):
                with worker_lock(root):
                    with self.assertRaisesRegex(RuntimeError, 'Another pilot worker'):
                        with worker_lock(root):
                            pass
                    raise RuntimeError('forced')
            with worker_lock(root):
                pass


if __name__ == '__main__':
    unittest.main()
