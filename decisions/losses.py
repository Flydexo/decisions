from __future__ import annotations

import torch


def confidence(probabilities, mask=None):
    """Normalized Shannon entropy over valid options only; singleton confidence is 1."""
    mask = torch.ones_like(probabilities, dtype=torch.bool) if mask is None else mask
    count = mask.sum(-1)
    if (count == 0).any():
        raise ValueError("Confidence needs at least one valid option")
    p = probabilities.masked_fill(~mask, 0)
    denominator = count.clamp_min(2).to(p.dtype).log()
    result = 1 + (p * p.clamp_min(1e-12).log()).sum(-1) / denominator
    return torch.where(count == 1, torch.ones_like(result), result.clamp(0, 1))


def reward(p, target, qtype, mask, weights, ordinal_order=None):
    log_score = (target * p.clamp_min(1e-12).log()).sum(-1).mean(-1)
    spherical = ((target * p).sum(-1) / p.square().sum(-1).sqrt().clamp_min(1e-12)).mean(-1)
    ranked_p, ranked_target = p, target
    if ordinal_order is not None:
        ranked_p = p.gather(-1, ordinal_order.unsqueeze(0).expand_as(p))
        ranked_target = target.gather(-1, ordinal_order.unsqueeze(0).expand_as(target))
    rps = ((qtype == 1) * (ranked_p.cumsum(-1) - ranked_target.cumsum(-1)).square().sum(-1)
           / mask.sum(-1)).sum(-1)
    return weights["log"] * log_score + weights["spherical"] * spherical - weights["rps"] * rps


def training_loss(logits, target, inputs, spans, training, ablation):
    if ablation.get("objective", "sampled_reward") == "cross_entropy":
        per_question = -(target * logits.log_softmax(-1)).sum(-1)
        return torch.stack([per_question[start:end].mean() for start, end, _ in spans]).mean()
    sigma, samples = training["sigma"], training["candidates"]
    if sigma <= 0 or samples < 2:
        raise ValueError("sigma must be positive and candidates must be >= 2")
    # Samples are actions, not a differentiable path through the distribution.
    candidates = logits.detach().unsqueeze(0) + sigma * torch.randn((samples, *logits.shape), device=logits.device)
    candidates = candidates.masked_fill(~inputs["marker_mask"].unsqueeze(0), -1e4)
    ordinal_order = None
    if inputs.get("option_labels") and ablation["reward_weights"]["rps"]:
        orders = []
        for keys in inputs["option_labels"]:
            # Score keys are canonical integer levels even when display order is
            # permuted. Sorting numeric choice keys is harmless: RPS masks them.
            positions = (sorted(range(len(keys)), key=lambda i: int(keys[i]))
                         if all(key.isdecimal() for key in keys) else list(range(len(keys))))
            orders.append(positions + list(range(len(keys), logits.shape[-1])))
        ordinal_order = torch.tensor(orders, dtype=torch.long, device=logits.device)
    with torch.no_grad():
        probabilities = candidates.softmax(-1)
        rewards = torch.stack([reward(probabilities[:, start:end, :width], target[start:end, :width].unsqueeze(0),
                                      inputs["qtype"][start:end], inputs["marker_mask"][start:end, :width],
                                      ablation["reward_weights"],
                                      ordinal_order[start:end, :width] if ordinal_order is not None else None)
                               for start, end, width in spans], dim=1)
        advantage = (rewards - rewards.mean(0, keepdim=True)) / (rewards.std(0, keepdim=True) + 1e-6)
    # REINFORCE differentiates log p(action | mean=logits). Differentiating a
    # zero-centered density through the sample reverses the reward gradient.
    normal = torch.distributions.Normal(logits.unsqueeze(0), candidates.new_tensor(sigma), validate_args=False)
    log_prob = normal.log_prob(candidates).masked_fill(~inputs["marker_mask"].unsqueeze(0), 0)
    return torch.stack([-(advantage[:, row, None, None] * log_prob[:, start:end, :width]).mean()
                        for row, (start, end, width) in enumerate(spans)]).mean()
