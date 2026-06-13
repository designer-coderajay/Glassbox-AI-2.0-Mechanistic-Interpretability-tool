# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.scaling
================
V5 throughput primitives (ROADMAP_V5_FOUNDATIONS.md Part 5).

This module holds the pure, deterministic *orchestration* logic for two of the
speedups — the parts that are architecture-agnostic and unit-testable:

  * ``plan_batches``      — length-sorted batching for the GPU batch path (item 1),
                            minimizing padding waste.
  * ``hierarchical_screen`` — layer-first screening that expands only the top
                            layers to head level (item 3), with an explicit
                            false-negative-risk report.

Out of scope here (torch / research, validated on real hardware):
  * GPU batched grads via vmap (item 1 execution),
  * KV-prefix sharing (item 2) — the roadmap flags this as unresolved: gradients
    must flow through the shared prefix, so naive cache reuse breaks the backward
    pass. Prototype before counting it; do NOT use for gradient attribution.

The screening accuracy caveat is real (Part 5 item 3 / Part 9.2 hole #2): a
circuit distributed across many weakly-contributing layers can evade a layer-level
screen. ``hierarchical_screen`` therefore reports the attribution mass it pruned
and raises a false-negative-risk flag — it is a speed heuristic to validate
against full search, not a compliance-path default.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Hashable, List, Tuple

__all__ = ["plan_batches", "hierarchical_screen"]


def plan_batches(
    items: List[Any],
    max_batch_size: int,
    *,
    length_of: Callable[[Any], int] = len,
    max_padded_tokens: int | None = None,
) -> List[List[Any]]:
    """Group items into padding-efficient batches (GPU batch path orchestration).

    Items are sorted by length so each batch pads to a similar length, then packed
    greedily up to ``max_batch_size`` and, optionally, a padded-token budget
    (``len(batch) * max_len_in_batch``). An item larger than the budget still goes
    in a batch by itself rather than being dropped.

    Args:
        items: The work items (e.g. prompts or token lists).
        max_batch_size: Hard cap on items per batch (must be >= 1).
        length_of: Maps an item to its length (default ``len``).
        max_padded_tokens: Optional cap on ``batch_size * max_len`` per batch.

    Returns:
        A list of batches (each a list of items); order within length bands is
        preserved, total items conserved.
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")
    if not items:
        return []

    order = sorted(range(len(items)), key=lambda i: length_of(items[i]))
    batches: List[List[Any]] = []
    cur: List[int] = []
    cur_maxlen = 0

    for i in order:
        length = length_of(items[i])
        prospective_max = max(cur_maxlen, length)
        prospective_padded = prospective_max * (len(cur) + 1)
        over_size = len(cur) >= max_batch_size
        over_tokens = (
            max_padded_tokens is not None and prospective_padded > max_padded_tokens
        )
        if cur and (over_size or over_tokens):
            batches.append([items[j] for j in cur])
            cur, cur_maxlen = [], 0
        cur.append(i)
        cur_maxlen = max(cur_maxlen, length)

    if cur:
        batches.append([items[j] for j in cur])
    return batches


def hierarchical_screen(
    layer_scores: Dict[int, float],
    head_to_layer: Dict[Hashable, int],
    *,
    layer_keep_frac: float = 0.5,
    min_layers: int = 1,
    risk_threshold: float = 0.10,
) -> Tuple[List[Hashable], List[Hashable], Dict[str, Any]]:
    """Layer-first circuit screening (Part 5 item 3).

    Keeps the top fraction of layers by ``|score|`` and marks for full head-level
    evaluation only the heads whose layer survived. Honest about its risk: reports
    the fraction of total layer ``|score|`` that was pruned, and flags
    ``false_negative_risk`` when that exceeds ``risk_threshold`` — i.e. when a
    meaningful share of the signal sits in pruned layers and the screen may miss a
    distributed circuit.

    Args:
        layer_scores: ``{layer_index: aggregate_attribution}``.
        head_to_layer: ``{head_key: layer_index}`` for every head.
        layer_keep_frac: Fraction of layers to keep (0-1).
        min_layers: Always keep at least this many layers.
        risk_threshold: Pruned-mass fraction above which to flag risk.

    Returns:
        ``(screened_heads, pruned_heads, report)``.
    """
    if not layer_scores:
        return [], list(head_to_layer), {
            "n_layers": 0, "n_layers_kept": 0,
            "n_heads_screened": 0, "n_heads_pruned": len(head_to_layer),
            "pruned_layer_mass_fraction": 0.0, "false_negative_risk": False,
            "note": "no layer scores provided",
        }

    by_score = sorted(layer_scores, key=lambda layer: abs(layer_scores[layer]), reverse=True)
    n_keep = max(min_layers, math.ceil(layer_keep_frac * len(by_score)))
    n_keep = min(n_keep, len(by_score))
    kept = set(by_score[:n_keep])
    pruned_layers = set(by_score[n_keep:])

    screened = [h for h, layer in head_to_layer.items() if layer in kept]
    pruned = [h for h, layer in head_to_layer.items() if layer not in kept]

    total = sum(abs(v) for v in layer_scores.values()) or 1.0
    pruned_mass = sum(abs(layer_scores[layer]) for layer in pruned_layers) / total
    risk = pruned_mass > risk_threshold

    report = {
        "n_layers": len(layer_scores),
        "n_layers_kept": len(kept),
        "n_heads_screened": len(screened),
        "n_heads_pruned": len(pruned),
        "pruned_layer_mass_fraction": round(pruned_mass, 4),
        "false_negative_risk": risk,
        "note": (
            "pruned layers hold a meaningful share of attribution mass; validate "
            "against full search before using in a compliance path"
            if risk else "pruned layers are low-signal; screening is likely safe"
        ),
    }
    return screened, pruned, report
