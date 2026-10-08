import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from datasets import ClassLabel, Features
from omegaconf import OmegaConf
from torch import nn

from decisions.adapters import Adapter
from decisions.archive import RangeReader
from decisions.checkpoints import read_checkpoint
from decisions.data import accepted, batches, dataset_configs, examples, load_stream, prepare_training_cache, sampled_examples
from decisions.evaluation import Metrics, Predictor
from decisions.losses import confidence, reward, training_loss
from decisions.model import DecisionModel
from decisions.schema import Unsupported, collate, prepare_request

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3
    mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        return {"input_ids": [4 + sum(map(ord, word)) % 120 for word in text.split()]}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embeddings = nn.Embedding(128, 16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embeddings(input_ids)))


def sample():
    return {"state": "some text", "questions": {
        "sentiment": {"type": "score", "instructions": "rate it", "criteria": ["bad", "ok", "good"],
                      "option_order": [2, 0, 1]},
        "yes": {"type": "noul", "instructions": "is it true?", "criteria": {}}},
        "targets": {"sentiment": {"0": 0.2, "1": 0.3, "2": 0.5}, "yes": {"false": 0., "true": 1.}}}


def config():
    cfg = OmegaConf.load(ROOT / "conf/config.yaml")
    del cfg.defaults
    del cfg.hydra
    cfg.model = OmegaConf.load(ROOT / "conf/model/default.yaml")
    cfg.model.nhead, cfg.model.dim_feedforward, cfg.model.num_layers = 4, 32, 1
    cfg.model.dropout = 0.1
    cfg.dataset = OmegaConf.load(ROOT / "conf/dataset/ag_news.yaml")
    cfg.ablation = OmegaConf.load(ROOT / "conf/ablation/cross_entropy.yaml")
    cfg.device, cfg.logging.enabled = "cpu", False
    cfg.training.epochs, cfg.training.max_steps = 2, 4
    cfg.training.batch_size, cfg.training.log_every = 2, 1
    cfg.training.checkpoint_every, cfg.training.cache_clear_every = 2, 2
    return cfg


class SchemaTests(unittest.TestCase):
    def test_batched_targets_follow_each_rows_option_permutation(self):
        rows = [sample(), {"state": "short", "questions": {"yes": {"type": "choice", "instructions": "pick",
                 "criteria": {"A": "a", "B": "b", "C": "c", "D": "d"}}},
                 "targets": {"yes": {"A": 0., "B": 1., "C": 0., "D": 0.}}}]
        inputs, target, spans = collate(rows, Tokenizer())
        self.assertEqual(spans, [(0, 2, 3), (2, 3, 4)])
        torch.testing.assert_close(target[0], torch.tensor([.5, .2, .3, 0.]))
        torch.testing.assert_close(target[1], torch.tensor([0., 1., 0., 0.]))
        self.assertEqual(inputs["question_ids"], ["sentiment", "yes", "yes"])

    def test_large_label_set_keeps_every_marker(self):
        row = {"state": "state " * 1000, "questions": {"q": {"type": "choice", "instructions": "classify",
               "criteria": [f"label_{i}" for i in range(77)]}}}
        result = prepare_request(row, Tokenizer())
        self.assertEqual(result["marker_mask"].sum(), 77)
        self.assertLessEqual(result["input_ids"].shape[1], 512)
        with self.assertRaises(Unsupported):
            prepare_request(row, Tokenizer(), strict=True)

    def test_strict_mode_retains_long_option_text_and_rejects_overflow(self):
        row = {"state": "", "questions": {"q": {"type": "choice", "instructions": "choose",
               "criteria": {"A": "word " * 80, "B": "other"}}}}
        strict = prepare_request(row, Tokenizer(), strict=True)
        truncated = prepare_request(row, Tokenizer(), strict=False)
        self.assertGreater(strict["input_ids"].shape[1], truncated["input_ids"].shape[1])
        with self.assertRaises(Unsupported):
            prepare_request(row, Tokenizer(), strict=True, max_len=32)

    def test_large_context_preserves_complete_options_before_truncating_state(self):
        row = {"state": "state " * 2000, "questions": {"q": {"type": "choice", "instructions": "choose",
               "criteria": {"A": "word " * 80, "B": "other"}}}}
        complete = prepare_request(dict(row, state=""), Tokenizer(), strict=True, max_len=1024)
        truncated_state = prepare_request(row, Tokenizer(), max_len=1024, preserve_options=True)
        self.assertEqual(truncated_state["marker_pos"].tolist(), complete["marker_pos"].tolist())
        self.assertEqual(truncated_state["input_ids"].shape[1], 1024)

    def test_long_structured_context_keeps_later_fields(self):
        row = {"state": {"old_file": "old " * 2000, "diff_hunk": "criticalfix"},
               "questions": {"q": {"type": "choice", "instructions": "choose", "criteria": ["yes", "no"]}}}
        tokenizer = Tokenizer()
        result = prepare_request(row, tokenizer, max_len=128, balanced_state=True)
        fix = tokenizer("criticalfix")["input_ids"][0]
        self.assertIn(fix, result["input_ids"][0].tolist())
        self.assertLessEqual(result["input_ids"].shape[1],128)

    def test_config_adapters_cover_actual_source_shapes(self):
        cases = {
            "ag_news": {"text": "news", "label": 1},
            "banking77": {"text": "card", "label": 1},
            "imdb": {"text": "film", "label": 1},
            "sst5": {"text": "great", "label": 4},
            "amazon_reviews_multi_en": {"text": "review", "label": 3},
            "yelp_review_full": {"text": "review", "label": 2},
            "boolq": {"passage": "passage", "question": "True?", "answer": False},
            "mnli": {"premise": "a", "hypothesis": "b", "label": 2},
            "dbpedia14": {"title": "entity", "content": "content", "label": 0},
            "arc_challenge": {"question": "science?", "choices": {"label": ["1", "2", "3"], "text": ["a", "b", "c"]}, "answerKey": "2"},
            "openbookqa": {"question_stem": "science?", "choices": {"label": ["A", "B"], "text": ["a", "b"]}, "answerKey": "A"},
            "commonsenseqa": {"question": "what?", "choices": {"label": ["A", "B"], "text": ["a", "b"]}, "answerKey": "B"},
            "aegis": {"prompt": "prompt", "response": "reply", "prompt_label": "safe", "response_label": "unsafe"},
            "consumer_finance": {"complaint_id": "1", "complaint_what_happened": "issue", "product": "Mortgage"},
            "trec": {"text": "Who is this?", "coarse_label": 3},
            "codereviewer": {"oldf": "code", "patch": "+line", "msg": "comment", "y": 0},
        }
        features = Features({"label": ClassLabel(names=["zero", "one", "two"])})
        for name, row in cases.items():
            with self.subTest(dataset=name):
                cfg = OmegaConf.to_container(OmegaConf.load(ROOT / f"conf/dataset/{name}.yaml"))
                canonical = Adapter(cfg["schema"], features)(row)
                _, target, _ = collate([canonical], Tokenizer())
                torch.testing.assert_close(target.sum(-1), torch.ones(len(target)))
                if name == "aegis":
                    self.assertEqual(len(target), 2)
                if name == "codereviewer":
                    self.assertEqual(target.argmax(-1).item(), 1)
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "conf/dataset/flakeflagger.yaml"))
        row = {path: 0 for path in cfg["schema"]["state"]["fields"].values()}
        row.update(flaky=1, test_name="test", project="project")
        canonical = Adapter(cfg["schema"])(row)
        self.assertNotIn("flaky", canonical["state"])
        self.assertNotIn("project", canonical["state"])

    def test_native_json_probabilities(self):
        s = sample()
        native = {"state": json.dumps(s["state"]), "questions": json.dumps(s["questions"]),
                  "gold": json.dumps({k: {"probabilities": v} for k, v in s["targets"].items()})}
        result = Adapter({"native": True})(native)
        self.assertEqual(result["targets"], s["targets"])


class LossTests(unittest.TestCase):
    def test_ordinal_reward_uses_level_order_after_option_permutation(self):
        p, target = torch.tensor([[[.8, .15, .05]]]), torch.tensor([[[1., 0., 0.]]])
        kind, mask = torch.tensor([1]), torch.ones((1, 3), dtype=torch.bool)
        weights = {"log": 0., "spherical": 0., "rps": 1.}
        canonical = reward(p, target, kind, mask, weights)
        permutation = [2, 0, 1]
        presented = reward(p[..., permutation], target[..., permutation], kind, mask, weights,
                           torch.tensor([[1, 2, 0]]))
        torch.testing.assert_close(canonical, presented)

    def test_entropy_certainty_is_distinct_from_probability_calibration(self):
        metrics = Metrics()
        p = torch.full((2, 2), .5)
        metrics.update(p, torch.tensor([[1., 0.], [0., 1.]]), torch.ones_like(p, dtype=torch.bool))
        result = metrics.result()
        self.assertEqual(result["entropy_confidence"], 0.)
        self.assertEqual(result["probability_ece"], 0.)
        self.assertEqual(result["entropy_accuracy_gap"], .5)

    def test_entropy_extremes_mask_and_singleton(self):
        p = torch.tensor([[.25, .25, .25, .25], [1., 0., 0., 0.], [.5, .5, 0., 0.], [1., 0., 0., 0.]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(confidence(p, mask), torch.tensor([0., 1., 0., 1.]))
        with self.assertRaises(ValueError):
            confidence(p[:1], torch.zeros_like(mask[:1]))

    def test_sampled_reward_gradient_increases_rewarded_answer(self):
        # A large candidate count makes the direction test robust to sampling noise.
        torch.manual_seed(21)
        logits = torch.zeros((1, 2), requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        target = torch.tensor([[0., 1.]])
        loss = training_loss(logits, target, {"qtype": torch.tensor([0]), "marker_mask": mask}, [(0, 1, 2)],
                             {"sigma": 1., "candidates": 2048},
                             {"objective": "sampled_reward", "reward_weights": {"log": 1., "spherical": .5, "rps": 1.}})
        loss.backward()
        updated = logits.detach() - .1 * logits.grad
        self.assertGreater(updated.softmax(-1)[0, 1].item(), .5)

    def test_padding_has_no_sampled_loss_gradient_and_rows_have_equal_weight(self):
        logits = torch.randn(3, 4, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
        target = mask.float() / mask.sum(-1, keepdim=True)
        loss = training_loss(logits.masked_fill(~mask, -1e4), target,
                             {"qtype": torch.tensor([0, 1, 0]), "marker_mask": mask}, [(0, 1, 2), (1, 3, 4)],
                             {"sigma": 1., "candidates": 8},
                             {"reward_weights": {"log": 1., "spherical": .5, "rps": 1.}})
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[~mask] == 0).all())
        settings = {"sigma": 1., "candidates": 8}
        ablation = {"reward_weights": {"log": 1., "spherical": .5, "rps": 1.}}
        logits = logits.detach().requires_grad_()
        noise = torch.randn(8, 3, 4)
        with patch("decisions.losses.torch.randn", side_effect=[noise, noise[:, :1, :2], noise[:, 1:]]):
            full = training_loss(logits, target, {"qtype": torch.tensor([0, 1, 0]), "marker_mask": mask},
                                 [(0, 1, 2), (1, 3, 4)], settings, ablation)
            row_one = training_loss(logits[:1, :2], target[:1, :2],
                                    {"qtype": torch.tensor([0]), "marker_mask": mask[:1, :2]},
                                    [(0, 1, 2)], settings, ablation)
            row_two = training_loss(logits[1:], target[1:],
                                    {"qtype": torch.tensor([1, 0]), "marker_mask": mask[1:]},
                                    [(0, 2, 4)], settings, ablation)
        reference = (row_one + row_two) / 2
        torch.testing.assert_close(full, reference)
        torch.testing.assert_close(torch.autograd.grad(full, logits)[0], torch.autograd.grad(reference, logits)[0])


class StreamingTests(unittest.TestCase):
    def test_stratified_samples_cover_classes_and_keep_partitions_disjoint(self):
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg['data'] = {'separate_validation':True,'validation_fraction':.2,
            'stratified_sampling':{'ag_news':{'field':'label','values':[0,1],'shuffle_buffer':0}}}
        dataset = dataset_configs(cfg)[0]
        seen = []
        class Stream:
            features = Features({'label':ClassLabel(names=['one','two'])})
            def __init__(self,value):self.value=value
            def __iter__(self):
                return iter({'text':f'example {i}','label':self.value} for i in range(200))
        def load(d,split):
            seen.append((split,d['source']['filters']))
            return Stream(d['source']['filters'][0][2])
        with patch('decisions.data.load_stream',side_effect=load):
            training=list(sampled_examples(dataset,'train',12))
            validation=list(sampled_examples(dataset,'validation',12))
            evaluation=list(sampled_examples(dataset,'eval',12))
        for rows in (training,validation,evaluation):
            labels=[max(r['targets']['label'],key=r['targets']['label'].get) for r in rows]
            self.assertEqual(labels.count('one'),6)
            self.assertEqual(labels.count('two'),6)
        self.assertFalse({r['state'] for r in training}&{r['state'] for r in validation})
        self.assertEqual([split for split,_ in seen],['train','train','train','train','test','test'])
        self.assertNotIn('filters',dataset['source'])

    def test_stratified_sampler_rejects_missing_classes(self):
        dataset={'name':'test','source':{},'stratified_sampling':{'field':'label','values':[0,1]}}
        with patch('decisions.data.examples',side_effect=lambda *a,**k:(r for r in [])):
            with self.assertRaisesRegex(ValueError,'class 0'):
                list(sampled_examples(dataset,'train',4))

    def test_loader_does_not_mutate_storage_option_provenance(self):
        cfg = {"name": "example", "source": {"format": "csv", "storage_options": {
            "client_kwargs": {"trust_env": True}}, "data_files": {"train": "train.csv"}}}
        original = copy.deepcopy(cfg)
        def library_call(*args, **kwargs):
            kwargs["storage_options"]["hf"] = {"endpoint": "https://huggingface.co", "token": None}
        with patch("decisions.data.load_dataset", side_effect=library_call):
            load_stream(cfg, "train")
        self.assertEqual(cfg, original)

    def test_hf_loader_is_always_streaming_and_only_requests_one_split(self):
        cfg = {"name": "example", "source": {"format": "json", "data_files": {"train": "train.jsonl", "test": "test.jsonl"}}}
        with patch("decisions.data.load_dataset") as loader:
            load_stream(cfg, "test")
        self.assertEqual(loader.call_args.kwargs["streaming"], True)
        self.assertEqual(loader.call_args.kwargs["data_files"], {"test": "test.jsonl"})

    def test_batches_are_lazy_and_keep_tail(self):
        consumed = []
        def rows():
            for i in range(5):
                consumed.append(i)
                yield i
        stream = batches(rows(), 2)
        self.assertEqual(next(stream), [0, 1])
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(list(stream), [[2, 3], [4]])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                list(batches([], bad))

    def test_hash_holdout_is_stable_disjoint_and_grouped(self):
        cfg = {"holdout": {"field": "project", "fraction": .2, "seed": 42}}
        for i in range(100):
            row = {"project": str(i), "test": 1}
            self.assertNotEqual(accepted(row, cfg, "train"), accepted(row, cfg, "eval"))
            self.assertEqual(accepted(row, cfg, "eval"), accepted(dict(row, test=2), cfg, "eval"))

    def test_finance_duplicate_narratives_stay_in_one_partition(self):
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "conf/dataset/consumer_finance.yaml"))
        cfg["holdout"]["validation_fraction"] = .1
        for i in range(100):
            row = {"complaint_id": str(i), "complaint_what_happened": f"Narrative {i}", "product": "Mortgage"}
            duplicate = dict(row, complaint_id=f"duplicate-{i}", product="Debt collection")
            roles = ["train", "validation", "eval"]
            original = [accepted(row, cfg, role) for role in roles]
            self.assertEqual(sum(original), 1)
            self.assertEqual(original, [accepted(duplicate, cfg, role) for role in roles])

    def test_archive_range_enforces_status_and_transfer_budget(self):
        reader = RangeReader("https://example.invalid/archive", 1000, max_transfer_bytes=100, block_size=50)
        response = SimpleNamespace(status_code=200, headers={}, raw=io.BytesIO(b"x" * 50))
        with patch.object(reader.session, "get") as get:
            get.return_value.__enter__.return_value = response
            with self.assertRaisesRegex(RuntimeError, "range"):
                reader.read(1)
        with self.assertRaises(ValueError):
            reader.read()
        reader = RangeReader("https://example.invalid/archive", 1000, max_transfer_bytes=10, block_size=50)
        with self.assertRaisesRegex(RuntimeError, "limit"):
            reader.read(1)


class ModelAndCheckpointTests(unittest.TestCase):
    def test_prenorm_layers_have_independent_reproducible_initializations(self):
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg['model'].update(norm_first=True, independent_init=True, num_layers=2)
        torch.manual_seed(42)
        model = DecisionModel(cfg['model'], cfg['ablation'], TinyEncoder())
        layers = model.transformer.layers
        self.assertTrue(all(layer.norm_first for layer in layers))
        self.assertFalse(torch.equal(layers[0].self_attn.in_proj_weight, layers[1].self_attn.in_proj_weight))
        self.assertFalse(torch.equal(layers[0].linear1.weight, layers[1].linear1.weight))
        torch.manual_seed(42)
        repeat = DecisionModel(cfg['model'], cfg['ablation'], TinyEncoder())
        self.assertTrue(all(torch.equal(value, repeat.head_state()[key]) for key, value in model.head_state().items()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.bert.parameters()))

    def test_feature_probe_preserves_training_mode_rng_and_scores_mixed_options(self):
        from decisions.diagnostics import feature_probe
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg['ablation'] = OmegaConf.to_container(OmegaConf.load(ROOT / 'conf/ablation/baseline.yaml'))
        cfg['model'].update(norm_first=True, independent_init=True)
        model = DecisionModel(cfg['model'], cfg['ablation'], TinyEncoder()).train()
        rng = torch.get_rng_state().clone()
        values = feature_probe(model, Tokenizer(), [sample()], torch.device('cpu'), cfg)
        self.assertTrue(model.training)
        self.assertFalse(model.bert.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(values['probe/questions'], 2)
        self.assertGreater(values['probe/reward_rps'], 0.)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in values.values()))
        from decisions.evaluation import preprocessing
        inputs, target, _ = collate([sample()], Tokenizer(), **preprocessing(cfg['model']))
        model.eval()
        with torch.no_grad():
            direct_nll = -(target * model(inputs).log_softmax(-1)).sum().item() / len(target)
        self.assertAlmostEqual(values['probe/nll'], direct_nll, places=6)

    def test_three_way_group_holdouts_are_disjoint(self):
        cfg = {"holdout": {"field": "project", "fraction": .2, "validation_fraction": .1, "seed": 42}}
        counts = {role: 0 for role in ("train", "validation", "eval")}
        for i in range(1000):
            row = {"project": str(i)}
            membership = [accepted(row, cfg, role) for role in counts]
            self.assertEqual(sum(membership), 1)
            for role, chosen in zip(counts, membership):
                counts[role] += chosen
                self.assertEqual(chosen, accepted(dict(row, test="another test"), cfg, role))
        self.assertTrue(all(counts.values()))

    def test_validation_reserves_duplicate_inputs_without_hashing_labels(self):
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg["data"] = {"separate_validation": True, "validation_fraction": .2}
        dataset = dataset_configs(cfg)[0]
        self.assertEqual(dataset["splits"]["validation"], "train")
        self.assertEqual(dataset["splits"]["eval"], "test")
        self.assertNotIn("validation", cfg["dataset"]["splits"])
        source = SimpleNamespace(features=Features({"label": ClassLabel(names=["one","two"])}))
        rows = [{"text": f"text {i}", "label": label} for i in range(100) for label in (0,1)]
        # A small real iterable stand-in makes each independent stream replayable.
        class Stream:
            features = source.features
            def __iter__(self):
                return iter(rows)
        with patch("decisions.data.load_stream", side_effect=lambda *a: Stream()):
            train_rows = list(examples(dataset, "train"))
            validation_rows = list(examples(dataset, "validation"))
        train_text = {r["state"] for r in train_rows}
        validation_text = {r["state"] for r in validation_rows}
        self.assertTrue(train_text and validation_text)
        self.assertFalse(train_text & validation_text)
        self.assertEqual(len(train_rows) + len(validation_rows), len(rows))

    def test_learning_rate_warmup_and_decay(self):
        from decisions.trainer import scheduled_learning_rate
        cfg = {"learning_rate": .01, "warmup_steps": 10, "decay_steps": 100,
               "min_learning_rate_ratio": .1}
        self.assertAlmostEqual(scheduled_learning_rate(cfg, 0), .001)
        self.assertAlmostEqual(scheduled_learning_rate(cfg, 9), .01)
        self.assertAlmostEqual(scheduled_learning_rate(cfg, 10), .01)
        self.assertAlmostEqual(scheduled_learning_rate(cfg, 100), .001)
        self.assertAlmostEqual(scheduled_learning_rate(cfg, 150), .001)

    def test_bounded_training_cache_never_reads_evaluation_and_reuses_samples(self):
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg["data"] = {"separate_validation": True, "train_pool_rows": 3}
        datasets = dataset_configs(cfg)
        reads = []
        def source(dataset, role, **kwargs):
            reads.append(role)
            for i in range(100):
                yield dict(sample(), _dataset=dataset["name"], _sample_id=i)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("decisions.data.examples", side_effect=source):
                counts = prepare_training_cache(cfg, datasets, root)
                self.assertEqual(counts, {"ag_news": 3})
                prepare_training_cache(cfg, datasets, root)
            self.assertEqual(reads, ["train"])
            cached = list(examples(datasets[0], "train", cache_dir=root))
            self.assertEqual([row["_sample_id"] for row in cached], [0,1,2])

    def test_inference_only_checkpoint_cannot_resume_training(self):
        from decisions.checkpoints import resume_checkpoint, save_checkpoint
        cfg = OmegaConf.to_container(config(), resolve=True)
        model = DecisionModel(cfg["model"], cfg["ablation"], TinyEncoder())
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, cfg, {"step":0}, torch.device('cpu'), include_optimizer=False)
            model.load_head(read_checkpoint(path)["head"])
            with self.assertRaisesRegex(ValueError, "inference-only"):
                resume_checkpoint(path, model, optimizer, cfg, [], torch.device('cpu'))

    def test_resume_preserves_consumed_time_budget(self):
        from decisions.trainer import train
        def loader(model_cfg, ablation, device):
            return DecisionModel(model_cfg, ablation, TinyEncoder()).to(device), Tokenizer()
        with tempfile.TemporaryDirectory() as directory, \
             patch("decisions.trainer.load_model", side_effect=loader), \
             patch("decisions.trainer.mixed_examples", side_effect=lambda *a, **k: iter([sample()] * 8)), \
             patch("decisions.trainer.validate"), patch("decisions.trainer.Logger"):
            directory = Path(directory)
            initial = config()
            initial.training.max_steps = 2
            initial.training.max_seconds = 3600
            train(initial, directory / 'initial')
            path = directory / 'initial/last.pt'
            saved = read_checkpoint(path)
            saved['progress']['training_elapsed_seconds'] = 3600.
            torch.save(saved, path)
            resumed = config()
            resumed.training.max_seconds = 3600
            resumed.resume = str(path)
            train(resumed, directory / 'resumed')
            result = read_checkpoint(directory / 'resumed/last.pt')
            self.assertEqual(result['progress']['step'], 2)
            for name in saved['head']:
                torch.testing.assert_close(result['head'][name], saved['head'][name], rtol=0, atol=0)

    def test_interrupted_optimizer_does_not_overwrite_safe_checkpoint(self):
        from decisions.trainer import train
        expected = {}
        def model_loader(model_cfg, ablation, device):
            model = DecisionModel(model_cfg, ablation, TinyEncoder()).to(device)
            expected.update({k: v.clone() for k, v in model.head_state().items()})
            return model, Tokenizer()
        def interrupt(optimizer, *args, **kwargs):
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].add_(1)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory, \
             patch("decisions.trainer.load_model", side_effect=model_loader), \
             patch("decisions.trainer.mixed_examples", side_effect=lambda *a, **k: iter([sample(), sample()])), \
             patch("decisions.trainer.Logger"), patch.object(torch.optim.AdamW, "step", new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                train(config(), directory)
            saved = read_checkpoint(Path(directory) / "last.pt")
            self.assertEqual(saved["progress"]["step"], 0)
            for name, value in expected.items():
                torch.testing.assert_close(saved["head"][name], value, rtol=0, atol=0)

    def test_encoder_is_frozen_for_every_architecture_ablation(self):
        cfg = OmegaConf.to_container(config().model)
        for flag in (None, "no_question_type", "no_transformer", "linear_scorer"):
            with self.subTest(ablation=flag):
                model = DecisionModel(cfg, {flag: True} if flag else {}, TinyEncoder())
                model.train()
                before = model.bert.embeddings.weight.detach().clone()
                inputs, target, _ = collate([sample()], Tokenizer())
                (-(model(inputs).log_softmax(-1) * target).sum()).backward()
                self.assertFalse(model.bert.training)
                self.assertTrue(all(p.grad is None for p in model.bert.parameters()))
                torch.testing.assert_close(model.bert.embeddings.weight, before)
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_resume_replays_stream_and_matches_uninterrupted_training_including_tail(self):
        from decisions.trainer import train
        def model_loader(model_cfg, ablation, device):
            return DecisionModel(model_cfg, ablation, TinyEncoder()).to(device), Tokenizer()
        rows = [sample() for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory, \
             patch("decisions.trainer.load_model", side_effect=model_loader), \
             patch("decisions.trainer.mixed_examples", side_effect=lambda *a, **k: iter(copy.deepcopy(rows))), \
             patch("decisions.trainer.validate"), patch("decisions.trainer.Logger"):
            directory = Path(directory)
            train(config(), directory / "full")
            interrupted = config()
            interrupted.training.max_steps = 2
            train(interrupted, directory / "part")
            resumed = config()
            resumed.resume = str(directory / "part/last.pt")
            train(resumed, directory / "resumed")
            full = read_checkpoint(directory / "full/last.pt")
            resumed = read_checkpoint(directory / "resumed/last.pt")
            self.assertEqual(full["progress"], resumed["progress"])
            self.assertEqual(full["progress"]["rows"], 7)
            for name in full["head"]:
                torch.testing.assert_close(full["head"][name], resumed["head"][name], rtol=0, atol=0)
            self.assertTrue(all(not name.startswith("bert.") for name in full["head"]))
            # A different evaluation sample must not compete with the old best score.
            from decisions.checkpoints import resume_checkpoint
            full["progress"]["best_accuracy"] = .9
            torch.save(full, directory / "full/last.pt")
            changed = OmegaConf.to_container(config(), resolve=True)
            changed["evaluation"]["max_rows"] = 16
            model, _ = model_loader(changed["model"], changed["ablation"], torch.device("cpu"))
            optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
            progress = resume_checkpoint(directory / "full/last.pt", model, optimizer, changed,
                                         full["config"]["dataset_configs"], torch.device("cpu"))
            self.assertEqual(progress["best_accuracy"], -1.)

    def test_predictor_returns_chosen_probability_and_entropy(self):
        predictor = Predictor.__new__(Predictor)
        predictor.config = {"model": {"max_len": 512, "head_max_len": 192, "option_max_len": 48}}
        predictor.tokenizer, predictor.device = Tokenizer(), torch.device("cpu")
        predictor.strict, predictor.question_batch_size, predictor.temperature = True, 1, 1.0
        predictor.model = lambda x: torch.zeros(x["marker_mask"].shape).masked_fill(~x["marker_mask"], -1e4)
        s = sample()
        result = predictor(s["state"], s["questions"])
        self.assertEqual(set(result["answers"]), set(s["questions"]))
        self.assertAlmostEqual(result["answers"]["yes"]["noul"], .5)
        self.assertAlmostEqual(result["answers"]["sentiment"]["score"], 1., places=6)
        for a in result["answers"].values():
            self.assertAlmostEqual(a["confidence"], a["chosen_probability"], places=6)
            self.assertAlmostEqual(a["entropy_confidence"], 0., places=6)


if __name__ == "__main__":
    unittest.main()
