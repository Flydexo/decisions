from __future__ import annotations

import torch
import logging
from torch import nn
from transformers import AutoModel, AutoTokenizer


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
                                       config["dropout"], batch_first=True),
            config["num_layers"], enable_nested_tensor=False)
        self.option_scorer = (nn.Linear(dimension, 1) if ablation.get("linear_scorer") else
                              nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension),
                                            nn.GELU(), nn.Linear(dimension, 1)))

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        return self

    def forward(self, inputs):
        # Ordinary no_grad tensors can be consumed by trainable layers; inference_mode
        # tensors cannot safely be saved for the head's backward pass.
        with torch.no_grad():
            features = self.bert(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"]).last_hidden_state
        if self.question_type is not None:
            features = features + self.question_type(inputs["qtype"]).unsqueeze(1)
        if self.transformer is not None:
            features = self.transformer(features, src_key_padding_mask=inputs["attention_mask"] == 0)
        positions = inputs["marker_pos"].unsqueeze(-1).expand(-1, -1, features.shape[-1])
        logits = self.option_scorer(features.gather(1, positions)).squeeze(-1)
        return logits.masked_fill(~inputs["marker_mask"], -1e4)

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
