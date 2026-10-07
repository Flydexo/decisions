from __future__ import annotations

import torch
import logging
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.checkpoint import checkpoint
from .precision import autocast_context, precision_dtype


def initialize_transformer_layers(encoder):
    """Give cloned layers independent draws, preserving each module's defaults."""
    for layer in encoder.layers:
        for module in layer.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.reset_parameters()
        for module in layer.modules():
            if isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                else:
                    for weight in (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight):
                        nn.init.xavier_uniform_(weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)


class DecisionModel(nn.Module):
    def __init__(self, config: dict, ablation: dict | None = None, encoder=None):
        super().__init__()
        ablation = ablation or {}
        precision_dtype(config)
        self.precision_settings = {"mixed_precision": config.get("mixed_precision", "fp32"),
                                   "autocast_cache_enabled": config.get("autocast_cache_enabled", False)}
        self.bert = encoder if encoder is not None else AutoModel.from_pretrained(
            config["encoder"], revision=config.get("revision"), attn_implementation="sdpa", dtype=torch.float32)
        train_encoder = config.get("train_encoder", False)
        last_layers = config.get("encoder_last_n_layers")
        if last_layers is not None and not train_encoder:
            raise ValueError("encoder_last_n_layers requires train_encoder=true")
        self.bert.requires_grad_(bool(train_encoder))
        if last_layers is not None:
            layers = getattr(self.bert, "layers", None)
            if (layers is None or isinstance(last_layers, bool) or not isinstance(last_layers, int)
                    or not 1 <= last_layers <= len(layers)):
                raise ValueError("encoder_last_n_layers must select existing encoder.layers")
            self.bert.requires_grad_(False)
            for layer in layers[-last_layers:]:
                layer.requires_grad_(True)
            if hasattr(self.bert, "final_norm"):
                self.bert.final_norm.requires_grad_(True)
        self.encoder_trainable = any(p.requires_grad for p in self.bert.parameters())
        self.head_checkpointing = config.get("gradient_checkpointing", False)
        if config.get("gradient_checkpointing", False):
            if not self.encoder_trainable:
                raise ValueError("Gradient checkpointing requires a trainable encoder")
            self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not self.encoder_trainable:
            self.bert.eval()
        dimension = self.bert.config.hidden_size
        self.question_type = None if ablation.get("no_question_type") else nn.Embedding(3, dimension)
        self.transformer = None if ablation.get("no_transformer") else nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dimension, config["nhead"], config["dim_feedforward"],
                                       config["dropout"], batch_first=True,
                                       norm_first=config.get("norm_first", False)),
            config["num_layers"], enable_nested_tensor=False)
        if self.transformer is not None and config.get("independent_init", False):
            initialize_transformer_layers(self.transformer)
        self.option_scorer = (nn.Linear(dimension, 1) if ablation.get("linear_scorer") else
                              nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension),
                                            nn.GELU(), nn.Linear(dimension, 1)))

    def train(self, mode=True):
        super().train(mode)
        if not self.encoder_trainable:
            self.bert.eval()
        return self

    def encode(self, inputs):
        # Ordinary no_grad tensors can be consumed by trainable layers; inference_mode
        # tensors cannot safely be saved for the head's backward pass.
        def forward():
            return self.bert(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"]).last_hidden_state
        if self.encoder_trainable:
            return forward()
        with torch.no_grad():
            return forward()

    def transform_features(self, features, inputs):
        if self.question_type is not None:
            features = features + self.question_type(inputs["qtype"]).unsqueeze(1)
        if self.transformer is not None:
            mask = inputs["attention_mask"] == 0
            if self.head_checkpointing and self.training and torch.is_grad_enabled():
                for layer in self.transformer.layers:
                    features = checkpoint(layer, features, src_key_padding_mask=mask,
                                          use_reentrant=False, preserve_rng_state=True)
                if self.transformer.norm is not None:
                    features = self.transformer.norm(features)
            else:
                features = self.transformer(features, src_key_padding_mask=mask)
        return features

    def score_features(self, features, inputs):
        positions = inputs["marker_pos"].unsqueeze(-1).expand(-1, -1, features.shape[-1])
        logits = self.option_scorer(features.gather(1, positions)).squeeze(-1)
        return logits.masked_fill(~inputs["marker_mask"], -1e4)

    def forward(self, inputs):
        with autocast_context(self.precision_settings, inputs["input_ids"].device):
            return self.score_features(self.transform_features(self.encode(inputs), inputs), inputs)

    def head_state(self, cpu=True):
        return {k: v.detach().cpu() if cpu else v.detach()
                for k, v in self.state_dict().items() if not k.startswith("bert.")}

    def load_head(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("bert.")}
        if set(state) != expected:
            raise ValueError("Checkpoint architecture does not match the trainable head")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("bert.") for key in result.missing_keys):
            raise ValueError("Incomplete trainable head in checkpoint")


def question_inputs(inputs, start, end):
    return {key: value[start:end] for key, value in inputs.items()}


def training_logits(model, inputs, question_microbatch_size=0):
    """Keep one joint loss while recomputing each bounded question forward."""
    count = len(inputs["qtype"])
    if not question_microbatch_size or count <= question_microbatch_size:
        return model(inputs)
    logits = []
    for start in range(0, count, question_microbatch_size):
        micro = question_inputs(inputs, start, start + question_microbatch_size)
        # Checkpoint the complete encoder+head forward so retaining logits for
        # the joint RLCD reward does not retain every question's activations.
        logits.append(checkpoint(model, micro, use_reentrant=False, preserve_rng_state=True))
    return torch.cat(logits)


def inference_logits(model, inputs, question_microbatch_size=0):
    count = len(inputs["qtype"])
    size = question_microbatch_size or count
    return torch.cat([model(question_inputs(inputs, start, start + size))
                      for start in range(0, count, size)])


def select_device(name="auto"):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process; run in a terminal with GPU access")
    return torch.device(name)


def load_model(config, ablation, device):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"], revision=config.get("revision"))
    for token in ("pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id"):
        if getattr(tokenizer, token) is None:
            raise ValueError(f"Encoder tokenizer needs {token}")
    model = DecisionModel(config, ablation).to(device)
    return model, tokenizer
