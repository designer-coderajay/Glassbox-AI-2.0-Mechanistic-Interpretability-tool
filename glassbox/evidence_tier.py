# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.evidence_tier
======================
V5 degradation ladder (ROADMAP_V5_FOUNDATIONS.md Part 6).

Every audit yields a report; what varies is the **evidence tier**, printed
on the report's first page. The two invariants this module enforces:

1. **Never silent** — every downgrade carries a stated reason, and the
   disclosure text names the tier and why it was reached.
2. **Never fabricate** — contradictory capability claims are rejected
   rather than reconciled (e.g. "exact patching verified" without weight
   access is a lie, not a configuration).

The ladder:

    A — causal-certified : circuit + exact patching + causal abstraction
    B — causal-screened  : first-order attribution, Hessian certificate clean
    C — behavioral       : black-box probing only
    D — descriptive      : system metadata + monitoring plan only

Pure stdlib; deterministic; fully unit-testable without any model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceTier",
    "TierSignals",
    "TierStep",
    "TierAssessment",
    "TierEngine",
]


class EvidenceTier(Enum):
    """The four evidence tiers, ordered strongest to weakest."""

    A_CAUSAL_CERTIFIED = ("A", "causal-certified")
    B_CAUSAL_SCREENED = ("B", "causal-screened")
    C_BEHAVIORAL = ("C", "behavioral")
    D_DESCRIPTIVE = ("D", "descriptive")

    @property
    def grade(self) -> str:
        """Stable single-letter grade for reports and APIs."""
        return self.value[0]

    @property
    def label(self) -> str:
        """Human-readable tier name."""
        return self.value[1]


_ORDER = [
    EvidenceTier.A_CAUSAL_CERTIFIED,
    EvidenceTier.B_CAUSAL_SCREENED,
    EvidenceTier.C_BEHAVIORAL,
    EvidenceTier.D_DESCRIPTIVE,
]


def _one_below(tier: EvidenceTier) -> EvidenceTier:
    idx = _ORDER.index(tier)
    return _ORDER[min(idx + 1, len(_ORDER) - 1)]


@dataclass(frozen=True)
class TierSignals:
    """Capability signals collected during an audit run.

    Attributes:
        has_weights: White-box access to model internals was available.
        counterfactual_valid: Whether the counterfactual gate passed.
            ``None`` means unverified, which is treated as NOT valid —
            unverified evidence is not evidence.
        hessian_reliable: Hessian certificate outcome (``None`` = not run).
        exact_patch_verified: Top-k units verified by exact activation
            patching (not just first-order approximation).
        causal_abstraction_tested: Interchange-intervention testing of the
            declared policy model was performed.
        behavioral_possible: A behavioral (black-box) probe was possible.
        sample_n: Number of samples behind the faithfulness estimates.
        min_sample_n: Hard floor below which results are underpowered
            (mirrors SampleSizeGate's blocking threshold).
    """

    has_weights: bool
    counterfactual_valid: Optional[bool]
    hessian_reliable: Optional[bool]
    exact_patch_verified: bool = False
    causal_abstraction_tested: bool = False
    behavioral_possible: bool = True
    sample_n: Optional[int] = None
    min_sample_n: int = 20

    def __post_init__(self) -> None:
        if self.exact_patch_verified and not self.has_weights:
            raise ValueError(
                "contradictory signals: exact_patch_verified=True requires "
                "weight access (has_weights=True). Refusing to assess a "
                "physically impossible capability claim."
            )
        if self.causal_abstraction_tested and not self.has_weights:
            raise ValueError(
                "contradictory signals: causal_abstraction_tested=True "
                "requires weight access (has_weights=True)."
            )


@dataclass(frozen=True)
class TierStep:
    """One recorded downgrade: from where, to where, and why."""

    from_tier: EvidenceTier
    to_tier: EvidenceTier
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "from": self.from_tier.grade,
            "to": self.to_tier.grade,
            "reason": self.reason,
        }


@dataclass
class TierAssessment:
    """The engine's verdict: a tier plus the full justification trail."""

    tier: EvidenceTier
    reasons: List[str] = field(default_factory=list)
    downgrades: List[TierStep] = field(default_factory=list)

    def disclosure_text(self) -> str:
        """One paragraph for the report's first page (Art. 13 honesty)."""
        head = (
            f"Evidence tier {self.tier.grade} ({self.tier.label})."
        )
        if not self.reasons:
            return (
                f"{head} Method: causal circuit analysis with exact "
                "patching verification and causal-abstraction testing."
            )
        return f"{head} Basis: " + " ".join(self.reasons)

    def summary_line(self) -> str:
        return (
            f"Tier {self.tier.grade} [{self.tier.label}] | "
            f"{len(self.downgrades)} downgrade(s)"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "tier": self.tier.grade,
            "label": self.tier.label,
            "reasons": list(self.reasons),
            "downgrades": [d.to_dict() for d in self.downgrades],
            "disclosure": self.disclosure_text(),
        }


class TierEngine:
    """Pure, deterministic mapping from signals to an evidence tier."""

    def assess(self, signals: TierSignals) -> TierAssessment:
        """Assess capability signals against the ladder.

        Args:
            signals: The audit run's capability signals.

        Returns:
            A :class:`TierAssessment` with tier, reasons, and the
            downgrade trail. Never raises for *weak* evidence — only for
            *contradictory* claims (handled in :class:`TierSignals`).
        """
        reasons: List[str] = []
        downgrades: List[TierStep] = []

        tier = self._capability_tier(signals, reasons, downgrades)
        tier = self._apply_sample_gate(signals, tier, reasons, downgrades)

        logger.debug("Tier assessment: %s (%d reasons)", tier.grade, len(reasons))
        return TierAssessment(tier=tier, reasons=reasons, downgrades=downgrades)

    # ------------------------------------------------------------------
    @staticmethod
    def _capability_tier(
        s: TierSignals, reasons: List[str], downgrades: List[TierStep]
    ) -> EvidenceTier:
        if not s.has_weights:
            if not s.behavioral_possible:
                reasons.append(
                    "No weight access and no behavioral probe available; "
                    "report is descriptive only (system metadata, logging "
                    "architecture, monitoring plan)."
                )
                return EvidenceTier.D_DESCRIPTIVE
            reasons.append(
                "No weight access: structural causal attribution is not "
                "possible; behavioral (black-box) evidence only."
            )
            return EvidenceTier.C_BEHAVIORAL

        if s.counterfactual_valid is not True:
            state = "failed" if s.counterfactual_valid is False else "unverified"
            reasons.append(
                f"Counterfactual validity {state}: causal attribution would "
                "be ungrounded; downgraded to behavioral evidence."
            )
            return EvidenceTier.C_BEHAVIORAL

        if s.exact_patch_verified and s.causal_abstraction_tested:
            return EvidenceTier.A_CAUSAL_CERTIFIED

        # Below A: explain exactly which certification is missing.
        if not s.exact_patch_verified:
            reasons.append(
                "Exact activation-patching verification not performed; "
                "first-order evidence only."
            )
        if not s.causal_abstraction_tested:
            reasons.append(
                "Causal-abstraction (interchange intervention) testing not "
                "performed."
            )

        if s.hessian_reliable is True:
            downgrades.append(TierStep(
                EvidenceTier.A_CAUSAL_CERTIFIED,
                EvidenceTier.B_CAUSAL_SCREENED,
                "Tier A certification incomplete; Hessian certificate clean, "
                "so first-order screening stands.",
            ))
            return EvidenceTier.B_CAUSAL_SCREENED

        state = "failed" if s.hessian_reliable is False else "not computed"
        reasons.append(
            f"Hessian reliability certificate {state}: first-order "
            "attribution cannot be trusted without it; downgraded to "
            "behavioral evidence."
        )
        downgrades.append(TierStep(
            EvidenceTier.B_CAUSAL_SCREENED,
            EvidenceTier.C_BEHAVIORAL,
            f"Hessian certificate {state}.",
        ))
        return EvidenceTier.C_BEHAVIORAL

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_sample_gate(
        s: TierSignals,
        tier: EvidenceTier,
        reasons: List[str],
        downgrades: List[TierStep],
    ) -> EvidenceTier:
        if s.sample_n is None or s.sample_n >= s.min_sample_n:
            return tier
        if tier in (EvidenceTier.C_BEHAVIORAL, EvidenceTier.D_DESCRIPTIVE):
            reasons.append(
                f"Sample size n={s.sample_n} below minimum "
                f"{s.min_sample_n}; estimates are underpowered."
            )
            return tier
        lower = _one_below(tier)
        reason = (
            f"Sample size n={s.sample_n} below minimum {s.min_sample_n}: "
            "faithfulness estimates are underpowered."
        )
        reasons.append(reason)
        downgrades.append(TierStep(tier, lower, reason))
        return lower
