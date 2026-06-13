# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.sequence_decision
==========================
V5 sequence decision value (ROADMAP_V5_FOUNDATIONS.md §2.1, multi-token form).

The single-token decision value reads one position's logits. For decision words
that span multiple tokens (" Approved", " Urgent"), the faithful quantity is the
teacher-forced sequence log-probability:

    log p(variant | prompt) = Σ_i  log softmax(logits at position p+i-1)[variant_i]

and the decision value pools variants with logsumexp via
``glassbox.decision.ResolvedDecision.value_from_scores``:

    D = logsumexp(scores over positive variants) − logsumexp(over negative)

Scope (honest): this computes the sequence decision *value* (a faithful measure
of which outcome the model prefers). It does NOT yet drive gradient attribution —
the attribution engine remains last-position (representative-token). Full
sequence-gradient attribution is the deeper research step (Roadmap §2.1, flagged
research-grade). Numbers from this module must be validated on a real model;
the pure aggregation is unit-tested, the model glue is lazy-imported.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

from glassbox.decision import DecisionFunctional

__all__ = [
    "teacher_forced_logprob",
    "sequence_decision_value",
    "model_scorer",
]


def _log_softmax_at(row: Sequence[float], idx: int) -> float:
    """Numerically stable log softmax of ``row`` evaluated at index ``idx``."""
    vals = [float(x) for x in row]
    m = max(vals)
    lse = m + math.log(sum(math.exp(v - m) for v in vals))
    return vals[idx] - lse


def teacher_forced_logprob(
    prompt_ids: Sequence[int],
    variant_ids: Sequence[int],
    forward_logits: Callable[[List[int]], Sequence[Sequence[float]]],
) -> float:
    """Teacher-forced ``log p(variant | prompt)``.

    Args:
        prompt_ids: Token ids of the prompt (non-empty).
        variant_ids: Token ids of one decision variant (>= 1 token).
        forward_logits: Callable mapping a token-id list to per-position logits
            ``[seq_len][vocab]`` (a 2D array/list, or anything index- and
            len-able). Position ``j`` holds the logits that predict token ``j+1``.

    Returns:
        Sum of per-token log-probabilities over the variant span.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty for teacher forcing")
    if not variant_ids:
        raise ValueError("variant_ids must contain at least one token")
    full = list(prompt_ids) + list(variant_ids)
    logits = forward_logits(full)
    p = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(variant_ids):
        total += _log_softmax_at(logits[p + i - 1], int(tok))
    return total


def sequence_decision_value(
    functional: DecisionFunctional,
    encode_variant: Callable[[str], Sequence[int]],
    prompt_ids: Sequence[int],
    forward_logits: Callable[[List[int]], Sequence[Sequence[float]]],
) -> float:
    """Sequence decision value D for a (possibly multi-token) verbalizer set.

    Resolves the functional's variants to token-id sequences (multi-token
    allowed), scores each by teacher-forced log-prob, and pools via the existing
    ``ResolvedDecision.value_from_scores`` (logsumexp positive − logsumexp negative).
    """
    resolved = functional.resolve(encode_variant)
    return resolved.value_from_scores(
        lambda ids: teacher_forced_logprob(prompt_ids, ids, forward_logits)
    )


def model_scorer(model):  # pragma: no cover - thin torch glue, validated on a real model
    """Bind teacher-forcing helpers to a TransformerLens model.

    Returns ``(encode_variant, forward_logits)``:
      * ``encode_variant(s)`` -> token-id list (no BOS), multi-token allowed;
      * ``forward_logits(token_ids)`` -> 2D numpy logits ``[seq_len][vocab]``.

    torch is imported lazily so this module is importable (and unit-testable)
    without it.
    """
    import torch

    def encode_variant(s: str) -> List[int]:
        return [int(i) for i in model.to_tokens(s, prepend_bos=False)[0]]

    def forward_logits(token_ids: List[int]):
        with torch.no_grad():
            logits = model(torch.tensor([token_ids]))
        return logits[0].float().cpu().numpy()

    return encode_variant, forward_logits
