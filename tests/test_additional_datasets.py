import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from decisions.adapters import Adapter
from decisions.data import accepted, dataset_configs, examples
from decisions.schema import collate
from test_pipeline import Tokenizer, config

ROOT = Path(__file__).resolve().parents[1]


def dataset(name):
    return OmegaConf.to_container(OmegaConf.load(ROOT / f"conf/dataset/{name}.yaml"), resolve=True)


class AdditionalDatasetTests(unittest.TestCase):
    def test_binary_labels_and_support_inputs(self):
        cases = [("enron_spam", {"text": "email", "label": 1}, "spam", True),
                 ("phishing_email", {"Email Text": "email", "Email Type": "Safe Email"}, "phishing", False),
                 ("phishing_email", {"Email Text": "email", "Email Type": "Phishing Email"}, "phishing", True)]
        for name, row, question, truth in cases:
            canonical = Adapter(dataset(name)["schema"])(row)
            self.assertEqual(canonical["targets"][question]["true"], float(truth))
            _, target, _ = collate([canonical], Tokenizer())
            self.assertEqual(target.argmax().item(), int(truth))
        row = dict(subject="subject", body="body", answer="SECRET answer", queue="Technical Support",
                   language="en", type="SECRET type", tag_1="SECRET tag")
        canonical = Adapter(dataset("customer_support")["schema"])(row)
        self.assertEqual(canonical["state"], {"subject": "subject", "body": "body"})
        self.assertNotIn("SECRET", str(canonical["questions"]))
        _, target, _ = collate([canonical], Tokenizer())
        self.assertEqual(target.shape, (1, 10))
        self.assertFalse(accepted(dict(row, language="de"), dataset("customer_support"), "train"))

    def test_marco_multiple_selected_and_unselected_passages(self):
        row = {"query_id": 7, "query": "query", "answers": ["SECRET"], "wellFormedAnswers": ["SECRET"],
               "passages": {"passage_text": ["first", "second", "third"], "is_selected": [1, 0, 1],
                            "url": ["SECRET"] * 3}}
        adapter = Adapter(dataset("ms_marco")["schema"])
        rows = list(adapter.iter_examples(row))
        self.assertEqual([r["id"] for r in rows], ["7:0", "7:1", "7:2"])
        self.assertEqual([r["targets"]["relevance"]["true"] for r in rows], [1., 0., 1.])
        for r in rows:
            self.assertEqual(set(r["state"]), {"query", "passage"})
            self.assertNotIn("SECRET", str(r["state"]) + str(r["questions"]))
        row["passages"]["is_selected"] = [0, 0, 0]
        self.assertEqual(len(list(adapter.iter_examples(row))), 3)
        row["passages"]["is_selected"] = [0]
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            list(adapter.iter_examples(row))

    def test_marco_expansion_is_lazy(self):
        row = {"query_id": 7, "query": "query", "passages": {
            "passage_text": ["first", "second"], "is_selected": [0, "invalid"]}}
        stream = Adapter(dataset("ms_marco")["schema"]).iter_examples(row)
        self.assertEqual(next(stream)["id"], "7:0")
        with self.assertRaisesRegex(ValueError, "Boolean"):
            next(stream)

    def test_all_passages_for_a_query_share_validation_partition(self):
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg["dataset"] = dataset("ms_marco")
        cfg["data"] = {"separate_validation": True, "validation_fraction": .2}
        source = dataset_configs(cfg)[0]
        rows = [{"query_id": i, "query": f"query {i}", "passages": {
            "passage_text": [f"passage {i}-{j}" for j in range(3)], "is_selected": [0, 1, 1]}}
                for i in range(100)]
        class Stream:
            features = None
            def __iter__(self):
                return iter(rows)
        with patch("decisions.data.load_stream", side_effect=lambda *a: Stream()):
            train = list(examples(source, "train"))
            validation = list(examples(source, "validation"))
        train_groups = {r["state"]["query"] for r in train}
        val_groups = {r["state"]["query"] for r in validation}
        self.assertTrue(train_groups and val_groups)
        self.assertFalse(train_groups & val_groups)
        self.assertEqual(len(train) + len(validation), 300)
        self.assertEqual(len(train), 3 * len(train_groups))
        self.assertEqual(len(validation), 3 * len(val_groups))
        self.assertEqual(source["splits"]["eval"], "validation")

    def test_content_holdouts_separate_duplicate_emails_and_tickets(self):
        for name, content_field in [("phishing_email", "Email Text"), ("customer_support", "body")]:
            cfg = OmegaConf.to_container(config(), resolve=True)
            cfg["dataset"] = dataset(name)
            cfg["data"] = {"separate_validation": True, "validation_fraction": .1}
            source = dataset_configs(cfg)[0]
            counts = {role: 0 for role in ("train", "validation", "eval")}
            for i in range(200):
                row = {content_field: f"body {i}", "Email Type": "Safe Email", "language": "en"}
                membership = [accepted(row, source, role) for role in counts]
                self.assertEqual(sum(membership), 1)
                duplicate = dict(row, id="other ID", subject="different subject")
                self.assertEqual(membership, [accepted(duplicate, source, role) for role in counts])
                for role, selected in zip(counts, membership):
                    counts[role] += selected
            self.assertTrue(all(counts.values()))

    def test_typed_all_uses_native_soft_targets(self):
        self.assertEqual(dataset("typed_decisions_all")["source"]["name"], "all")
        self.assertEqual(dataset("typed_decisions")["source"]["name"], "customer_service")
        row = {"state": {"context": "context"}, "questions": {"q": {"type": "noul", "instructions": "decide"}},
               "gold": {"q": {"probabilities": {"false": .25, "true": .75}}}, "factors": "SECRET"}
        result = Adapter(dataset("typed_decisions_all")["schema"])(row)
        self.assertEqual(result["targets"]["q"], {"false": .25, "true": .75})
        self.assertNotIn("factors", result)


if __name__ == "__main__":
    unittest.main()
