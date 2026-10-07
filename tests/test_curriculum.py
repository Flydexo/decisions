import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from decisions.curriculum import ordered_sources, replay_buffer, with_replay
from decisions.losses import training_loss


ROOT = Path(__file__).resolve().parents[1]


class CurriculumTests(unittest.TestCase):
    def test_plan_uses_full_splits_and_all_validation_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "not-created"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_rtx4090_curriculum.py"),
                                     "--plan", "--run-dir", str(run_dir)], capture_output=True,
                                    text=True, check=True)
            plan = json.loads(result.stdout)
            stages = plan["stages"]
            self.assertEqual(len(stages), 21)
            self.assertEqual(stages[0]["dataset"], "arc_challenge")
            self.assertEqual(stages[-1]["dataset"], "consumer_finance")
            self.assertEqual([s["dataset"] for s in stages], plan["evaluation_datasets_after_each_stage"])
            self.assertEqual([s["published_train_rows"] for s in stages],
                             sorted(s["published_train_rows"] for s in stages))
            self.assertNotIn("ms_marco", plan["config"]["mixture"])
            self.assertNotIn("commitpackft", plan["config"]["mixture"])
            self.assertFalse(plan["config"]["data"]["prepare_train_cache"])
            self.assertIsNone(plan["config"]["data"]["train_pool_rows"])
            self.assertEqual(plan["config"]["training"]["decay_steps"], 0)
            self.assertEqual(plan["config"]["training"]["validation_role"], "validation")
            self.assertFalse(run_dir.exists())

    def test_duplicate_or_unknown_sources_cannot_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ordered_sources(["boolq", "boolq"])
        with self.assertRaisesRegex(ValueError, "Missing published"):
            ordered_sources(["unknown"])

    def test_replay_uses_only_prior_training_rows_and_caches_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = {"name": "earlier", "source": {"path": "test"}}
            settings = {"replay_rows_per_source": 2, "shuffle_buffer": 0}
            saved = [{"_dataset": "earlier", "state": "a"},
                     {"_dataset": "earlier", "state": "b"}]
            with patch("decisions.curriculum.sampled_examples", return_value=iter(saved)) as sample:
                self.assertEqual(replay_buffer(dataset, temporary, settings, 42), saved)
                self.assertEqual(replay_buffer(dataset, temporary, settings, 42), saved)
                sample.assert_called_once()
            current = [{"_dataset": "current", "state": str(i)} for i in range(6)]
            combined, count = with_replay(current, {"earlier": saved}, .25)
            self.assertEqual(combined[:6], current)
            self.assertEqual(count, 2)
            self.assertTrue(all(row["_dataset"] == "earlier" for row in combined[6:]))

    def test_supervised_anchor_has_gradient_when_sampled_rewards_saturate(self):
        logits = torch.tensor([[100., -100.]], requires_grad=True)
        target = torch.tensor([[0., 1.]])
        inputs = {"marker_mask": torch.tensor([[True, True]]), "qtype": torch.tensor([0])}
        settings = {"sigma": 1., "candidates": 32, "auxiliary_cross_entropy_weight": .1}
        ablation = {"objective": "sampled_reward", "reward_weights": {"log": 1., "spherical": .5, "rps": 1.}}
        torch.manual_seed(1)
        loss = training_loss(logits, target, inputs, [(0, 1, 2)], settings, ablation)
        loss.backward()
        self.assertGreater(loss.item(), 10)
        self.assertGreater(logits.grad.abs().sum().item(), .1)


if __name__ == "__main__":
    unittest.main()
