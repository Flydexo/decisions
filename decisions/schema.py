from __future__ import annotations

import json
import re
from typing import Any

import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class Unsupported(ValueError):
    """A complete request cannot fit the configured model."""


def field(row: dict, path: str) -> Any:
    value = row
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def render(template: str, row: dict) -> str:
    return re.sub(r"\{([\w.]+)\}", lambda m: str(field(row, m[1])), template)


def labels(question: dict) -> list[str]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        result = list(criteria) if isinstance(criteria, (dict, list)) else []
        result = [str(x) for x in result]
    elif kind == "score":
        result = [str(i) for i in range(len(criteria))]
    elif kind == "noul":
        result = ["false", "true"]
    else:
        raise ValueError(f"Unknown question type {kind!r}")
    if not result or len(set(result)) != len(result):
        raise ValueError("Options must be nonempty and unique")
    return result


def presented_labels(question: dict) -> list[str]:
    result = labels(question)
    order = question.get("option_order", list(range(len(result))))
    if sorted(order) != list(range(len(result))):
        raise ValueError("option_order must be a permutation")
    return [result[i] for i in order]


def option_texts(question: dict) -> list[str]:
    kind, criteria = question["type"], question.get("criteria")
    serialize = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    if kind == "choice":
        criteria = criteria if isinstance(criteria, dict) else {str(v): None for v in criteria}
        result = [str(k) if v in (None, "") else f"{k}: {serialize(v)}" for k, v in criteria.items()]
    elif kind == "score":
        result = [f"level {i}: {serialize(v)}" for i, v in enumerate(criteria)]
    else:
        criteria = {str(k).lower(): v for k, v in (criteria or {}).items()}
        display = question.get("labels", {"false": "false", "true": "true"})
        result = [f"{display[key]}: {serialize(criteria.get(key) or default)}" for key, default in
                  [("false", "no, the statement does not hold"), ("true", "yes, the statement holds")]]
    presented_labels(question)  # Validate before touching token positions.
    return [result[i] for i in question.get("option_order", range(len(result)))]


def prepare_request(schema: dict, tokenizer, max_len=512, head_max_len=192,
                    option_max_len=48, strict=False, preserve_options=False, balanced_state=False) -> dict:
    """One sequence per question. Strict inference never truncates the payload."""
    encode = lambda text: tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    clean = lambda text: str(text) if strict else str(text).replace(tokenizer.mask_token, " ")
    state = schema["state"]
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    state_ids = encode(clean(state_text))
    items, question_ids, option_labels = [], [], []
    for qid, question in schema["questions"].items():
        instruction = question["instructions"]
        if not isinstance(instruction, str):
            instruction = json.dumps(instruction, ensure_ascii=False)
        head = encode(clean(f"{question['type']} question: {instruction}"))
        options = [[tokenizer.mask_token_id] + encode(" " + clean(text))
                   for text in option_texts(question)]
        full_length = len(head) + sum(map(len, options)) + len(state_ids) + 4
        if strict and full_length > max_len:
            raise Unsupported(f"Question {qid!r} needs {full_length} tokens; capacity is {max_len}")
        if not strict and not (preserve_options and len(head) + sum(map(len, options)) + 4 <= max_len):
            # Preserve every option marker and some text even for large label sets.
            min_head = min(16, len(head))
            if 2 * len(options) + min_head + 4 > max_len:
                raise Unsupported(f"{len(options)} options cannot fit without dropping options")
            budget = min(max_len - 4, max(head_max_len, min(6, option_max_len + 1) * len(options) + min_head))
            per_option = max(2, min(option_max_len + 1, (budget - min_head) // len(options)))
            options = [tokens[:per_option] for tokens in options]
            head = head[:max(min_head, budget - sum(map(len, options)))]
        ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
        markers = []
        for tokens in options:
            markers.append(len(ids))
            ids.extend(tokens)
        ids.append(tokenizer.sep_token_id)
        room = max_len - len(ids) - 1
        selected = state_ids if strict else (state_ids[-room:] if isinstance(state, list) and room > 0
                                            else state_ids[:max(room, 0)])
        if not strict and balanced_state and isinstance(state, dict) and len(state_ids) > room and room > 0:
            fragments = [encode(clean(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}\n"))
                         for key, value in state.items()]
            # Water-fill the available context: short fields keep their full
            # contents; long fields share the remainder instead of dropping the
            # later diff hunk or response. This never incorporates target labels.
            budgets = [0] * len(fragments)
            remaining = room
            active = list(range(len(fragments)))
            while active and remaining:
                share = max(1, remaining // len(active))
                survivors = []
                for index in active:
                    take = min(share, len(fragments[index]) - budgets[index], remaining)
                    budgets[index] += take
                    remaining -= take
                    if budgets[index] < len(fragments[index]):
                        survivors.append(index)
                active = survivors
            selected = [token for fragment, budget in zip(fragments, budgets) for token in fragment[:budget]]
        ids.extend(selected)
        ids.append(tokenizer.sep_token_id)
        if len(ids) > max_len or len(markers) != len(labels(question)):
            raise Unsupported("Request cannot fit without losing options")
        items.append((ids, markers, QTYPES[question["type"]]))
        question_ids.append(qid)
        option_labels.append(presented_labels(question))
    if not items:
        raise ValueError("At least one question is required")
    seq_len, width = max(len(x[0]) for x in items), max(len(x[1]) for x in items)
    result = {"input_ids": torch.full((len(items), seq_len), tokenizer.pad_token_id, dtype=torch.long),
              "attention_mask": torch.zeros((len(items), seq_len), dtype=torch.long),
              "marker_pos": torch.zeros((len(items), width), dtype=torch.long),
              "marker_mask": torch.zeros((len(items), width), dtype=torch.bool),
              "qtype": torch.tensor([x[2] for x in items]),
              "question_ids": question_ids, "option_labels": option_labels}
    for i, (ids, markers, _) in enumerate(items):
        result["input_ids"][i, :len(ids)] = torch.tensor(ids)
        result["attention_mask"][i, :len(ids)] = 1
        result["marker_pos"][i, :len(markers)] = torch.tensor(markers)
        result["marker_mask"][i, :len(markers)] = True
    return result


def collate(rows: list[dict], tokenizer, **options):
    if not rows:
        raise ValueError("At least one dataset row is required")
    inputs, targets, spans, offset = [], [], [], 0
    for row in rows:
        item = prepare_request(row, tokenizer, **options)
        target = torch.zeros(item["marker_mask"].shape)
        for i, (qid, keys) in enumerate(zip(item["question_ids"], item["option_labels"])):
            distribution = row["targets"][qid]
            values = torch.tensor([distribution[key] for key in keys], dtype=torch.float32)
            if not torch.isfinite(values).all() or (values < 0).any() or not torch.isclose(
                    values.sum(), torch.tensor(1.0), atol=1e-4, rtol=0):
                raise ValueError(f"{qid}: invalid target probabilities")
            target[i, :len(keys)] = values
        n, k = target.shape
        spans.append((offset, offset + n, k))
        offset += n
        inputs.append(item)
        targets.append(target)
    seq_len = max(x["input_ids"].shape[1] for x in inputs)
    width = max(x["marker_pos"].shape[1] for x in inputs)
    result = {}
    for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask"):
        size = seq_len if key in ("input_ids", "attention_mask") else width
        pad = tokenizer.pad_token_id if key == "input_ids" else 0
        result[key] = torch.cat([torch.nn.functional.pad(x[key], (0, size - x[key].shape[1]), value=pad)
                                 for x in inputs])
    result["qtype"] = torch.cat([x["qtype"] for x in inputs])
    for key in ("question_ids", "option_labels"):
        result[key] = [v for x in inputs for v in x[key]]
    target = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1])) for x in targets])
    return result, target, spans


def to_device(inputs, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
