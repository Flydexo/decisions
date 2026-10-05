import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from torch import nn
from transformers import ModernBertConfig, ModernBertModel


class TestTokenizer:
    mask_token = "[MASK]"
    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3

    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        tokens = [4 + sum(map(ord, word)) % 120 for word in text.split()]
        if truncation:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


def sample_dataset():
    questions = [
        {"shared": {"type": "choice", "instructions": "Pick an answer",
                    "criteria": ["no", "yes"], "option_order": [1, 0]}},
        {"shared": {"type": "score", "instructions": "Choose a score",
                    "criteria": ["low", "medium", "high"], "option_order": [2, 0, 1]},
         "truth": {"type": "noul", "instructions": "Is this true?",
                   "criteria": {}, "option_order": [1, 0]},
         "category": {"type": "choice", "instructions": "Choose a category",
                      "criteria": ["a", "b", "c", "d"]}},
        {"shared": {"type": "choice", "instructions": "Choose a label",
                    "criteria": ["a", "b", "c"]}},
    ]
    probabilities = [
        {"shared": {"no": 0.3, "yes": 0.7}},
        {"shared": {"0": 0.2, "1": 0.3, "2": 0.5},
         "truth": {"false": 0.6, "true": 0.4},
         "category": {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}},
        {"shared": {"a": 0.2, "b": 0.3, "c": 0.5}},
    ]
    return Dataset.from_list([
        {"state": json.dumps(state), "questions": json.dumps(question),
         "gold": json.dumps({key: {"probabilities": value} for key, value in gold.items()})}
        for state, question, gold in zip(
            ["short", "long state " * 20, "tail " * 4], questions, probabilities
        )
    ])


class TrainingBatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        notebook = json.loads((Path(__file__).resolve().parents[1] / "model.ipynb").read_text())
        self.sources = ["".join(cell["source"]) for cell in notebook["cells"]]

        def encoder(*args, **kwargs):
            config = ModernBertConfig(
                vocab_size=128, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4,
                max_position_embeddings=512, local_attention=16, pad_token_id=0,
                bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
            )
            config._attn_implementation = "sdpa"
            return ModernBertModel(config)

        self.ns = {"torch": torch, "nn": nn,
                   "AutoModel": SimpleNamespace(from_pretrained=encoder),
                   "model_id": "test", "device": torch.device("cpu"),
                   "tokenizer": TestTokenizer()}
        for index in (2, 3, 4, 6, 8, 9, 10):
            exec(compile(self.sources[index], f"model.ipynb:cell{index}", "exec"), self.ns)
        self.ds = sample_dataset()

    def batch(self, indices):
        return self.ns["prepare_dataset_batch"](self.ds, indices, self.ns["tokenizer"])

    def old_row_loss(self, candidates, target, qtype, mask):
        # Reference: the single-row objective used before dataset batching.
        with torch.no_grad():
            rewards = self.ns["reward"](candidates.softmax(-1), target.unsqueeze(0), qtype, mask)
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        normal = torch.distributions.Normal(candidates.new_zeros(()), candidates.new_tensor(1.0))
        log_prob = normal.log_prob(candidates).masked_fill(~mask.unsqueeze(0), 0.0)
        return -(advantage[:, None, None] * log_prob).mean()

    def test_padding_targets_and_row_boundaries(self):
        inputs, target, spans = self.batch(range(3))
        self.assertEqual(spans, [(0, 1, 2), (1, 4, 4), (4, 5, 3)])
        self.assertEqual(inputs["question_ids"].count("shared"), 3)
        for index, (start, end, width) in enumerate(spans):
            single = self.ns["prepare_laya_inputs"](
                self.ns["ds_to_schema"](self.ds, index), self.ns["tokenizer"]
            )
            single_target = self.ns["ds_to_target"](self.ds, index, single)
            length = single["input_ids"].shape[1]
            for key in ("input_ids", "attention_mask"):
                torch.testing.assert_close(inputs[key][start:end, :length], single[key])
                self.assertTrue((inputs[key][start:end, length:] == 0).all())
            for key in ("marker_pos", "marker_mask"):
                torch.testing.assert_close(inputs[key][start:end, :width], single[key])
                self.assertTrue((inputs[key][start:end, width:] == 0).all())
            torch.testing.assert_close(target[start:end, :width], single_target)
            self.assertTrue((target[start:end, width:] == 0).all())
            torch.testing.assert_close(inputs["qtype"][start:end], single["qtype"])

    def test_loss_and_gradients_equal_mean_of_independent_rows(self):
        for indices in ([0], [0, 1, 2]):
            with self.subTest(rows=len(indices)):
                inputs, target, spans = self.batch(indices)
                raw = torch.randn(8, *target.shape, requires_grad=True)
                candidates = raw.masked_fill(~inputs["marker_mask"].unsqueeze(0), -1e4)
                loss = self.ns["batch_candidate_loss"](
                    candidates, target, inputs["qtype"], inputs["marker_mask"], spans, 1.0
                )
                reference = torch.stack([
                    self.old_row_loss(candidates[:, start:end, :width], target[start:end, :width],
                                      inputs["qtype"][start:end], inputs["marker_mask"][start:end, :width])
                    for start, end, width in spans
                ]).mean()
                torch.testing.assert_close(loss, reference)
                gradient = torch.autograd.grad(loss, raw, retain_graph=True)[0]
                reference_gradient = torch.autograd.grad(reference, raw)[0]
                torch.testing.assert_close(gradient, reference_gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient[:, ~inputs["marker_mask"]] == 0).all())

    def test_training_keeps_tail_batch_and_encoder_frozen(self):
        model = self.ns["DecisionModel"](16, 4, 1, 32, 0.0)
        encoder_before = {name: p.detach().clone() for name, p in model.bert.named_parameters()}
        head_before = model.option_scorer.layers[-1].weight.detach().clone()
        batches = []
        original = self.ns["prepare_dataset_batch"]

        def record_batch(ds, indices, tokenizer):
            batches.append(list(indices))
            return original(ds, indices, tokenizer)

        self.ns.update(dec=model, ds={"train": self.ds}, batch_size=2,
                       max_steps=None, log_every=1, prepare_dataset_batch=record_batch)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(self.sources[11], self.ns)
        self.assertEqual(batches, [[0, 1], [2]])
        self.assertEqual(self.ns["num_steps"], 2)
        self.assertIn("rows 3/3", output.getvalue())
        self.assertFalse(torch.equal(head_before, model.option_scorer.layers[-1].weight))
        self.assertFalse(model.bert.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, p in model.bert.named_parameters():
            torch.testing.assert_close(p, encoder_before[name], rtol=0, atol=0)
        for parameter, state in self.ns["optimizer"].state.items():
            self.assertTrue(parameter.requires_grad)
            self.assertEqual(state["step"].item(), 2)

    def test_invalid_batch_sizes_and_empty_batches_are_rejected(self):
        for size in (0, -1, 1.5, True):
            with self.subTest(batch_size=size):
                self.ns["batch_size"] = size
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    exec(self.sources[11], self.ns)
        with self.assertRaisesRegex(ValueError, "At least one dataset row"):
            self.batch([])


if __name__ == "__main__":
    unittest.main()
