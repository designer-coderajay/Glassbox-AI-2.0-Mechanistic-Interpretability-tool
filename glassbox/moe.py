# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.moe
============
V5 Mixture-of-Experts partition (ROADMAP_V5_FOUNDATIONS.md Part 4.4).

MoE is the immediate test of the graph-partition abstraction (Part 2.2): the
units are the experts (plus the router). Expert-level attribution is *coarser but
more causal* than head-level — routing is literally a discrete causal decision —
and no compliance tool on the market has any MoE story at all.

This module provides the two pure, testable pieces:
  * ``moe_units``        — enumerate the expert/router UnitSpecs for an MoE config;
  * ``expert_attribution`` — combine the router's weights with per-expert
    contributions into a ranked expert attribution.

The real Mixtral adapter (forward hooks reading router logits + expert outputs on
a `transformers` MoE model) is torch and validated against the model; it
implements the same ``AuditableModel`` protocol and is gated by the same
conformance suite — which is the whole point: a new architecture is "write the
adapter, pass conformance, ship."
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

from glassbox.auditable import UnitSpec

__all__ = ["moe_units", "expert_attribution"]


def moe_units(
    n_layers: int,
    n_experts_per_layer: int,
    *,
    include_router: bool = True,
) -> List[UnitSpec]:
    """Enumerate the MoE computation-graph partition as UnitSpecs.

    Each layer contributes one router unit (the discrete routing decision) and one
    unit per expert. This is the partition the attribution math runs over for an
    MoE model.

    Args:
        n_layers: Number of MoE layers.
        n_experts_per_layer: Experts per layer (e.g. 8 for Mixtral).
        include_router: Whether to emit a router unit per layer.

    Returns:
        Ordered list of UnitSpecs (router before its experts, per layer).
    """
    if n_layers < 1 or n_experts_per_layer < 1:
        raise ValueError("n_layers and n_experts_per_layer must be >= 1")
    units: List[UnitSpec] = []
    for layer in range(n_layers):
        if include_router:
            units.append(UnitSpec(name=f"L{layer}.router", layer=layer, kind="router", index=0))
        for e in range(n_experts_per_layer):
            units.append(UnitSpec(name=f"L{layer}.expert{e}", layer=layer, kind="expert", index=e))
    return units


def expert_attribution(
    routing_weights: Dict[Hashable, float],
    expert_contributions: Dict[Hashable, float],
) -> List[Tuple[Hashable, float]]:
    """Rank experts by causal attribution = routing weight × contribution.

    An expert only affects the output to the extent the router actually sends the
    token to it (the routing weight) AND the expert moves the decision value (its
    contribution). Multiplying the two gives the expert's causal share; experts
    with zero routing weight contribute nothing regardless of capacity.

    Args:
        routing_weights: ``{expert_key: weight}`` from the router (0 if not routed).
        expert_contributions: ``{expert_key: Δ decision value}`` per expert.

    Returns:
        ``[(expert_key, attribution), ...]`` sorted by ``|attribution|`` desc.
        Only experts present in ``expert_contributions`` are scored; a missing
        routing weight is treated as 0 (not routed → no causal effect).
    """
    scored: List[Tuple[Hashable, float]] = []
    for expert, contrib in expert_contributions.items():
        weight = float(routing_weights.get(expert, 0.0))
        scored.append((expert, weight * float(contrib)))
    scored.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return scored
