# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.causal_abstraction
===========================
V5 causal-abstraction certificate (ROADMAP_V5_FOUNDATIONS.md Part 2.4 — the moat).

The strongest compliance statement is not "these heads correlate with the
decision" but "this circuit *implements* the declared decision policy." That is
tested by interchange interventions: set the high-level causal variable to a
value, perform the corresponding intervention on the low-level circuit, and check
the model's output matches what the abstraction predicts. The fraction that match
is the interchange-intervention accuracy (IIA).

Honest scope (Part 9.1: "open research at production scale; per-customer causal
models are consulting-shaped work"): generating the interchange trials requires a
declared causal model and real interventions (torch). This module computes the
**certificate** from those trials — the pure, testable part — and states the IIA
honestly. A high IIA is what makes a result tier-A (causal-certified) eligible.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

__all__ = ["interchange_accuracy", "certify_abstraction"]

Trial = Tuple[Any, Any]  # (predicted_by_abstraction, observed_from_model)


def interchange_accuracy(trials: Sequence[Trial]) -> float:
    """Interchange-intervention accuracy: fraction of trials where the model's
    observed output matches the abstraction's prediction.

    Args:
        trials: ``[(predicted, observed), ...]``. Equality is ``==`` so labels,
            tokens, or ids all work as long as both sides use the same type.

    Returns:
        IIA in [0, 1].
    """
    n = len(trials)
    if n == 0:
        raise ValueError("interchange_accuracy needs at least one trial")
    matches = sum(1 for predicted, observed in trials if predicted == observed)
    return matches / n


def certify_abstraction(
    trials: Sequence[Trial],
    *,
    threshold: float = 0.80,
) -> Dict[str, Any]:
    """Build a causal-abstraction certificate from interchange trials.

    The circuit is certified as implementing the declared policy when IIA meets
    ``threshold``. This is the gate for tier A (causal-certified): without it, a
    circuit is at best first-order screened (tier B) — see glassbox.evidence_tier.

    Returns:
        ``{interchange_accuracy, n_trials, threshold, certified, tier_eligible,
        note}`` — vault-ready, and honest about what a pass means.
    """
    iia = interchange_accuracy(trials)
    certified = iia >= threshold
    return {
        "interchange_accuracy": round(iia, 4),
        "n_trials": len(trials),
        "threshold": threshold,
        "certified": certified,
        "tier_eligible": "A (causal-certified)" if certified else "below A",
        "note": (
            "circuit reproduces the declared policy under interchange "
            "interventions" if certified else
            "interchange accuracy below threshold; the circuit does not reliably "
            "implement the declared policy — do not claim causal certification"
        ),
    }
