# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.features
=================
V5 feature-level units via SAEs / transcoders (ROADMAP_V5_FOUNDATIONS.md Part 4.4,
Phase C — the field's direction).

Where head-level faithfulness is weak, the partition refines to *features*:
directions in a sparse-autoencoder / transcoder basis. Features fit the same
``AuditableModel`` partition (``UnitSpec(kind="feature")``) — the math is
unchanged, only the units are finer.

Honest scope (Part 9.1: "feature-level attribution at frontier scale is open
research"): training the SAE/transcoder and reading feature directions is torch
and lives in ``sae_attribution.py`` (the seed). This module provides the
partition and the sparsity-aware attribution primitive — the pure, testable
pieces. A defining property of SAE features is sparsity: only the few features
*active* on a token can carry causal effect, so attribution skips the rest.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

from glassbox.auditable import UnitSpec

__all__ = ["feature_units", "sparse_feature_attribution"]


def feature_units(layer_feature_counts: Dict[int, int]) -> List[UnitSpec]:
    """Enumerate SAE feature units as the partition.

    Args:
        layer_feature_counts: ``{layer_index: n_features}``.

    Returns:
        Ordered UnitSpecs with ``kind="feature"`` (by layer, then feature index).
    """
    if any(c < 0 for c in layer_feature_counts.values()):
        raise ValueError("feature counts must be >= 0")
    units: List[UnitSpec] = []
    for layer in sorted(layer_feature_counts):
        for f in range(layer_feature_counts[layer]):
            units.append(UnitSpec(name=f"L{layer}.f{f}", layer=layer, kind="feature", index=f))
    return units


def sparse_feature_attribution(
    activations: Dict[Hashable, float],
    contributions: Dict[Hashable, float],
    *,
    eps: float = 0.0,
) -> List[Tuple[Hashable, float]]:
    """Rank *active* features by attribution = activation × contribution.

    Only features whose activation magnitude exceeds ``eps`` are scored — an
    inactive feature (the overwhelming majority, by SAE sparsity) cannot have a
    causal effect on this token regardless of its decoder direction.

    Args:
        activations: ``{feature_key: activation}`` for this input.
        contributions: ``{feature_key: Δ decision value per unit activation}``.
        eps: Activation magnitude floor for "active".

    Returns:
        ``[(feature_key, attribution), ...]`` for active features, sorted by
        ``|attribution|`` descending.
    """
    scored: List[Tuple[Hashable, float]] = []
    for feat, act in activations.items():
        a = float(act)
        if abs(a) <= eps:
            continue
        scored.append((feat, a * float(contributions.get(feat, 0.0))))
    scored.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return scored
