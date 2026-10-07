from __future__ import annotations

import torch
import logging
from torch import nn
from transformers import AutoModel, AutoTokenizer


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
        self.bert = encoder if encoder is not None else AutoModel.from_pretrained(
            config["encoder"], revision=config.get("revision"), attn_implementation="sdpa", dtype=torch.float32)
        self.bert.requires_grad_(False).eval()
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
        self.bert.eval()
        return self

    def encode(self, inputs):
        # Ordinary no_grad tensors can be consumed by trainable layers; inference_mode
        # tensors cannot safely be saved for the head's backward pass.
        with torch.no_grad():
            return self.bert(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"]).last_hidden_state

    def transform_features(self, features, inputs):
        if self.question_type is not None:
            features = features + self.question_type(inputs["qtype"]).unsqueeze(1)
        if self.transformer is not None:
            features = self.transformer(features, src_key_padding_mask=inputs["attention_mask"] == 0)
        return features

    def score_features(self, features, inputs):
        positions = inputs["marker_pos"].unsqueeze(-1).expand(-1, -1, features.shape[-1])
        logits = self.option_scorer(features.gather(1, positions)).squeeze(-1)
        return logits.masked_fill(~inputs["marker_mask"], -1e4)

    def forward(self, inputs):
        return self.score_features(self.transform_features(self.encode(inputs), inputs), inputs)

    def head_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("bert.")}

    def load_head(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("bert.")}
        if set(state) != expected:
            raise ValueError("Checkpoint architecture does not match the trainable head")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("bert.") for key in result.missing_keys):
            raise ValueError("Incomplete trainable head in checkpoint")


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
