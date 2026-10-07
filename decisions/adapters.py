from __future__ import annotations

import json
import re

from .schema import field, labels, render


class Adapter:
    """Declarative row-to-request mapping; labels come from config/metadata, never a scan."""

    def __init__(self, config: dict, features=None):
        self.config = config
        self.features = features

    def iter_examples(self, original: dict):
        """Expand aligned source lists lazily into independent decision rows."""
        explode = self.config.get("explode")
        if not explode:
            yield self(original)
            return
        columns = {name: field(original, path) for name, path in explode["fields"].items()}
        if not columns or any(not isinstance(values, (list, tuple)) for values in columns.values()):
            raise ValueError("Exploded fields must be nonempty mappings of source lists")
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Exploded source lists must have matching lengths")
        identity = original.get(self.config.get("id_field", "id"))
        for index in range(next(iter(lengths))):
            row = dict(original)
            row.update({name: values[index] for name, values in columns.items()})
            canonical = self(row)
            if identity is not None:
                canonical["id"] = f"{identity}:{index}"
            yield canonical

    def __call__(self, original: dict) -> dict:
        cfg = self.config
        if cfg.get("native"):
            parse = lambda value: json.loads(value) if isinstance(value, str) else value
            gold = parse(original[cfg.get("gold_field", "gold")])
            return {"state": parse(original["state"]), "questions": parse(original["questions"]),
                    "targets": {k: v["probabilities"] for k, v in gold.items()}}
        row = dict(original)
        for name, rule in cfg.get("derive", {}).items():
            value = field(row, rule["field"])
            if "regex" in rule:
                match = re.search(rule["regex"], str(value))
                if match is None:
                    raise ValueError(f"Cannot extract {name} from {value!r}")
                value = match.group(rule.get("group", 1))
            if "mapping" in rule:
                value = rule["mapping"][str(value)]
            row[name] = value
        state_rule = cfg.get("state", {"literal": ""})
        if "field" in state_rule:
            state = field(row, state_rule["field"])
        elif "fields" in state_rule:
            state = {name: field(row, path) for name, path in state_rule["fields"].items()}
        elif "template" in state_rule:
            state = render(state_rule["template"], row)
        else:
            state = state_rule.get("literal", "")
        questions, targets = {}, {}
        for qid, spec in cfg["questions"].items():
            question = {"type": spec["type"], "instructions": render(spec["instructions"], row)}
            if spec.get("options"):
                opt = spec["options"]
                texts = field(row, opt["texts"])
                keys = field(row, opt["keys"]) if opt.get("keys") else [str(i) for i in range(len(texts))]
                if len(keys) != len(texts) or len(set(map(str, keys))) != len(keys):
                    raise ValueError("Option texts and unique keys must align")
                question["criteria"] = dict(zip(map(str, keys), texts))
            elif spec.get("labels_from_feature"):
                if self.features is None:
                    raise ValueError("This mapping requires ClassLabel metadata")
                feature = self.features[spec["labels_from_feature"]]
                if not hasattr(feature, "names"):
                    raise ValueError("Provide explicit criteria for labels without ClassLabel metadata")
                question["criteria"] = list(feature.names)
            else:
                question["criteria"] = spec.get("criteria", {})
            if "labels" in spec:
                question["labels"] = spec["labels"]
            keys = labels(question)
            target_spec = spec["target"]
            mode = target_spec.get("mode", "index")
            if mode == "comment_presence":
                # Match CodeReviewer's published read_review_examples labeling.
                value = 1 if row.get("msg", "") else int(row.get("y", 0))
                mode = "index"
            else:
                value = field(row, target_spec["field"])
            if mode != "probabilities" and "mapping" in target_spec:
                value = target_spec["mapping"][str(value)]
            if mode == "probabilities":
                distribution = {str(k): float(v) for k, v in value.items()}
            elif mode == "boolean":
                if not isinstance(value, bool) and value not in (0, 1):
                    raise ValueError(f"Expected Boolean target, got {value!r}")
                distribution = {"false": float(not value), "true": float(bool(value))}
            else:
                if mode == "label":
                    key = str(value)
                    if key not in keys:
                        raise ValueError(f"Label {key!r} is absent from configured options")
                elif mode == "index":
                    index = int(value) - target_spec.get("offset", 0)
                    if index < 0 or index >= len(keys):
                        raise ValueError(f"Target index {index} is out of bounds")
                    key = keys[index]
                else:
                    raise ValueError(f"Unknown target mode {mode}")
                distribution = {k: float(k == key) for k in keys}
            if set(distribution) != set(keys):
                raise ValueError("Target keys must match every option")
            questions[qid], targets[qid] = question, distribution
        return {"state": state, "questions": questions, "targets": targets,
                "id": row.get(cfg.get("id_field", "id"))}
