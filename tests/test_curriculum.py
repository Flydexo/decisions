import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from decisions.curriculum import ordered_sources


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


if __name__ == "__main__":
    unittest.main()
