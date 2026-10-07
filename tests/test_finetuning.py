import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import ModernBertConfig, ModernBertModel

from decisions.checkpoints import load_checkpoint_model, read_checkpoint, resume_checkpoint, save_checkpoint
from decisions.losses import training_loss
from decisions.model import DecisionModel, inference_logits, training_logits
from decisions.schema import collate
from decisions.trainer import backward_microbatches, build_optimizer, train
from decisions.precision import build_scaler, optimizer_step
from test_pipeline import Tokenizer, TinyEncoder, config, sample


def modern_encoder():
    cfg = ModernBertConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=128,
                          local_attention=16, global_attn_every_n_layers=2, pad_token_id=0,
                          bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    cfg._attn_implementation = "sdpa"
    return ModernBertModel(cfg)


class FinetuningTests(unittest.TestCase):
    def test_bf16_checkpointed_forward_keeps_weights_moments_and_loss_fp32(self):
        cfg = self.settings()
        cfg["model"]["mixed_precision"] = "bf16"
        cfg["ablation"] = {"objective": "sampled_reward", "reward_weights": {"log": 1., "spherical": .5, "rps": 1.}}
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).train()
        inputs, target, spans = collate([sample()], Tokenizer())
        logits = training_logits(model, inputs, 1)
        self.assertEqual(logits.dtype, torch.bfloat16)
        loss = training_loss(logits, target, inputs, spans, cfg["training"], cfg["ablation"])
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        opt = build_optimizer(model, cfg["training"])
        _, updated = optimizer_step(opt, list(model.parameters()), 1., build_scaler(cfg["model"], torch.device("cpu")))
        self.assertTrue(updated)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(v.dtype == torch.float32 for state in opt.state.values()
                            for k, v in state.items() if k in {"exp_avg", "exp_avg_sq"}))
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(inference_logits(model, inputs, 1)).all())

    def test_fp16_scaler_checkpoint_restores_scale_and_growth_counter(self):
        cfg = self.settings()
        cfg["model"]["mixed_precision"] = "fp16"
        cfg["dataset_configs"] = []
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).train()
        opt = build_optimizer(model, cfg["training"])
        opt._decision_scaler = build_scaler(cfg["model"], torch.device("cpu"))
        opt._decision_scaler.load_state_dict({"scale": 128., "growth_factor": 2., "backoff_factor": .5,
                                            "growth_interval": 2000, "_growth_tracker": 13})
        backward_microbatches(model, Tokenizer(), [[sample()]], torch.device("cpu"), cfg, opt._decision_scaler)
        _, updated = optimizer_step(opt, list(model.parameters()), 1., opt._decision_scaler)
        self.assertTrue(updated)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, opt, cfg, {"step": 1}, torch.device("cpu"))
            restored = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder())
            new_opt = build_optimizer(restored, cfg["training"])
            new_opt._decision_scaler = build_scaler(cfg["model"], torch.device("cpu"))
            resume_checkpoint(path, restored, new_opt, cfg, [], torch.device("cpu"))
            self.assertEqual(new_opt._decision_scaler.state_dict(), opt._decision_scaler.state_dict())

    def test_fp16_nonfinite_gradients_skip_update_and_reduce_scale(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.AdamW([parameter], foreach=False)
        scaler = build_scaler({"mixed_precision": "fp16"}, torch.device("cpu"))
        scaler.scale(parameter.sum()).backward()
        parameter.grad.fill_(float("inf"))
        norm, updated = optimizer_step(opt, [parameter], 1., scaler)
        self.assertFalse(updated)
        self.assertIsNone(norm)
        self.assertEqual(parameter.item(), 1.)
        self.assertEqual(scaler.get_scale(), 512.)

    def settings(self):
        cfg = OmegaConf.to_container(config(), resolve=True)
        cfg["model"].update(train_encoder=True, gradient_checkpointing=True, dropout=0.)
        cfg["training"].update(encoder_learning_rate=1e-5, question_microbatch_size=1,
                              gradient_accumulation_steps=2, optimizer_foreach=False,
                              clear_cache_between_microbatches=True)
        return cfg

    def test_full_checkpointed_encoder_receives_gradients_and_updates(self):
        cfg = self.settings()
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).train()
        self.assertTrue(model.bert.training)
        self.assertTrue(model.bert.is_gradient_checkpointing)
        self.assertTrue(all(p.requires_grad for p in model.bert.parameters()))
        before = model.bert.final_norm.weight.detach().clone()
        optimizer = build_optimizer(model, cfg["training"])
        backward_microbatches(model, Tokenizer(), [[sample()], [sample()]], torch.device("cpu"), cfg)
        self.assertGreater(model.bert.embeddings.tok_embeddings.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.bert.layers[0].mlp.Wi.weight.grad.abs().sum().item(), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.bert.final_norm.weight))
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["head", "encoder"])
        self.assertEqual(optimizer.param_groups[1]["lr"], 1e-5)

    def test_last_layer_freezes_embeddings_and_earlier_layers(self):
        cfg = self.settings()
        cfg["model"]["encoder_last_n_layers"] = 1
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).train()
        before = model.bert.layers[0].mlp.Wi.weight.detach().clone()
        optimizer = build_optimizer(model, cfg["training"])
        backward_microbatches(model, Tokenizer(), [[sample()]], torch.device("cpu"), cfg)
        self.assertIsNone(model.bert.embeddings.tok_embeddings.weight.grad)
        self.assertIsNone(model.bert.layers[0].mlp.Wi.weight.grad)
        self.assertIsNotNone(model.bert.layers[-1].mlp.Wi.weight.grad)
        optimizer.step()
        torch.testing.assert_close(before, model.bert.layers[0].mlp.Wi.weight, rtol=0, atol=0)

    def test_question_microbatches_preserve_joint_rlcd_reward_and_gradients(self):
        cfg = self.settings()
        cfg["ablation"] = {"objective": "sampled_reward", "reward_weights": {"log": 1., "spherical": .5, "rps": 1.}}
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).eval()
        reference = copy.deepcopy(model)
        inputs, target, spans = collate([sample()], Tokenizer())
        torch.manual_seed(123)
        full = training_loss(reference(inputs), target, inputs, spans, cfg["training"], cfg["ablation"])
        full.backward()
        torch.manual_seed(123)
        micro = training_loss(training_logits(model, inputs, 1), target, inputs, spans, cfg["training"], cfg["ablation"])
        micro.backward()
        torch.testing.assert_close(full, micro, atol=2e-6, rtol=2e-5)
        for (_, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(inference_logits(model, inputs, 1), reference(inputs), atol=1e-6, rtol=1e-5)

    def test_accumulation_weights_short_tail_by_rows(self):
        cfg = self.settings()
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).eval()
        reference = copy.deepcopy(model)
        rows = [sample(), sample(), sample()]
        inputs, target, spans = collate(rows, Tokenizer())
        full = training_loss(reference(inputs), target, inputs, spans, cfg["training"], cfg["ablation"])
        full.backward()
        value = backward_microbatches(model, Tokenizer(), [rows[:2], rows[2:]], torch.device("cpu"), cfg)
        self.assertAlmostEqual(value, full.item(), places=6)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)

    def test_checkpoint_retains_encoder_for_resume_and_inference(self):
        cfg = self.settings()
        cfg["dataset_configs"] = []
        model = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder()).train()
        optimizer = build_optimizer(model, cfg["training"])
        backward_microbatches(model, Tokenizer(), [[sample()]], torch.device("cpu"), cfg)
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, cfg, {"step": 1}, torch.device("cpu"))
            saved = read_checkpoint(path)
            self.assertEqual(saved["format_version"], 2)
            restored = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder())
            opt = build_optimizer(restored, cfg["training"])
            resume_checkpoint(path, restored, opt, cfg, [], torch.device("cpu"))
            for p, q in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            best = Path(directory) / "best.pt"
            save_checkpoint(best, model, optimizer, cfg, {"step": 1}, torch.device("cpu"), include_optimizer=False)
            inference = DecisionModel(cfg["model"], cfg["ablation"], modern_encoder())
            load_checkpoint_model(inference, read_checkpoint(best))
            for p, q in zip(model.parameters(), inference.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            saved.pop("encoder")
            with self.assertRaisesRegex(ValueError, "trained encoder"):
                load_checkpoint_model(inference, saved)
            changed = copy.deepcopy(cfg)
            changed["training"]["gradient_accumulation_steps"] = 3
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                resume_checkpoint(path, restored, opt, changed, [], torch.device("cpu"))

    def test_unfrozen_accumulated_resume_matches_full_run(self):
        def loader(model_cfg, ablation, device):
            return DecisionModel(model_cfg, ablation, TinyEncoder()).to(device), Tokenizer()
        cfg = config()
        cfg.model.train_encoder = True
        cfg.training.gradient_accumulation_steps = 2
        cfg.training.question_microbatch_size = 1
        cfg.training.encoder_learning_rate = 1e-5
        cfg.training.optimizer_foreach = False
        cfg.training.clear_cache_between_microbatches = True
        rows = [sample() for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory, \
             patch("decisions.trainer.load_model", side_effect=loader), \
             patch("decisions.trainer.mixed_examples", side_effect=lambda *a, **k: iter(copy.deepcopy(rows))), \
             patch("decisions.trainer.validate"), patch("decisions.trainer.Logger"):
            root = Path(directory)
            train(cfg, root / "full")
            partial = copy.deepcopy(cfg)
            partial.training.max_steps = 1
            train(partial, root / "part")
            resumed = copy.deepcopy(cfg)
            resumed.resume = str(root / "part/last.pt")
            train(resumed, root / "resumed")
            full = read_checkpoint(root / "full/last.pt")
            result = read_checkpoint(root / "resumed/last.pt")
            self.assertEqual(full["progress"], result["progress"])
            for section in ("head", "encoder"):
                for name in full[section]:
                    torch.testing.assert_close(full[section][name], result[section][name], rtol=0, atol=0)

    def test_bounded_training_pools_are_prepared_before_model_allocation(self):
        cfg = config()
        cfg.model.train_encoder = True
        cfg.training.max_steps = 1
        cfg.data = {"prepare_train_cache": True, "train_pool_rows": 2}
        prepared = []
        def prepare(configuration, datasets, path):
            prepared.append(path)
            return {"ag_news": 2}
        def loader(model_cfg, ablation, device):
            self.assertEqual(len(prepared), 1)
            return DecisionModel(model_cfg, ablation, TinyEncoder()).to(device), Tokenizer()
        with tempfile.TemporaryDirectory() as directory, \
             patch("decisions.trainer.prepare_training_cache", side_effect=prepare), \
             patch("decisions.trainer.load_model", side_effect=loader), \
             patch("decisions.trainer.mixed_examples", side_effect=lambda *a, **k: iter([sample(), sample()])), \
             patch("decisions.trainer.validate"), patch("decisions.trainer.Logger"):
            train(cfg, directory)
            saved = read_checkpoint(Path(directory) / "last.pt")
            self.assertEqual(prepared, [Path(directory).resolve() / "training_samples"])
            self.assertEqual(saved["config"]["data"]["train_cache_dir"], str(prepared[0]))


if __name__ == "__main__":
    unittest.main()
