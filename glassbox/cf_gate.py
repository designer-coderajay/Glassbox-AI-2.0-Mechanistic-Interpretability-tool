# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.cf_gate
================
V5 counterfactual verification gate (ROADMAP_V5_FOUNDATIONS.md §3.3) —
"the step everyone skips."

A generated counterfactual is admitted into attribution only if it:

  (a) still parses as the same task        → otherwise TASK_DRIFT
  (b) is alignable to the clean prompt     → otherwise ALIGNMENT
  (c) actually moves the decision value    → otherwise NULL_EFFECT
      (a counterfactual that changes nothing measures nothing)

Invalid candidates are **discarded and reported** — the discard counts go
into the technical file. Silence is how tools lie; this module exists so
Glassbox structurally cannot.

Model-free by design: the ΔD measurement is injected as a callable, so the
gate itself is pure logic and runs in any environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DiscardReason",
    "CandidateCF",
    "GateConfig",
    "Discarded",
    "GateResult",
    "CounterfactualGate",
]


class DiscardReason(Enum):
    """Why a counterfactual candidate was rejected."""

    TASK_DRIFT = "task_drift"
    ALIGNMENT = "alignment"
    NULL_EFFECT = "null_effect"
    MEASUREMENT_FAILED = "measurement_failed"


@dataclass(frozen=True)
class CandidateCF:
    """One counterfactual candidate awaiting verification.

    Attributes:
        text: The counterfactual prompt text (for reporting).
        strategy: Which generation strategy produced it (name_swap,
            antonym, semantic_negation, random_token, ...).
        tokens: Tokenized form; required for the structural checks.
    """

    text: str
    strategy: str
    tokens: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class GateConfig:
    """Gate thresholds.

    Attributes:
        min_len_ratio: Candidate/clean token-length ratio floor below
            which the candidate no longer resembles the task.
        max_len_ratio: Ratio ceiling above which it no longer resembles
            the task.
        noise_floor: Minimum |ΔD| for the candidate to count as having
            an effect. Values at the floor are kept (>=).
        require_alignment: When True, candidate token length must equal
            the clean length so position-wise patching is well-defined.
        min_valid: Minimum surviving candidates for the gate to report
            ``sufficient=True``.
    """

    min_len_ratio: float = 0.5
    max_len_ratio: float = 2.0
    noise_floor: float = 0.05
    require_alignment: bool = True
    min_valid: int = 1


@dataclass(frozen=True)
class Discarded:
    """A rejected candidate with its reason and diagnostic detail."""

    candidate: CandidateCF
    reason: DiscardReason
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "strategy": self.candidate.strategy,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass
class GateResult:
    """Outcome of gating a batch of candidates."""

    valid: List[CandidateCF]
    discarded: List[Discarded]
    config: GateConfig = field(default_factory=GateConfig)

    @property
    def sufficient(self) -> bool:
        """True when enough candidates survived for a grounded audit."""
        return len(self.valid) >= self.config.min_valid

    def discard_report(self) -> Dict[str, object]:
        """JSON-safe summary for the technical file (Annex IV §4 honesty)."""
        by_reason: Dict[str, int] = {}
        for d in self.discarded:
            by_reason[d.reason.value] = by_reason.get(d.reason.value, 0) + 1
        return {
            "n_candidates": len(self.valid) + len(self.discarded),
            "n_valid": len(self.valid),
            "sufficient": self.sufficient,
            "discarded_by_reason": by_reason,
            "discarded": [d.to_dict() for d in self.discarded],
        }


class CounterfactualGate:
    """Verifies counterfactual candidates before they touch attribution."""

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()

    def evaluate(
        self,
        clean_tokens: Sequence[int],
        candidates: Sequence[CandidateCF],
        measure_delta: Callable[[CandidateCF], float],
    ) -> GateResult:
        """Gate a batch of candidates.

        Args:
            clean_tokens: Tokenized clean prompt (the reference).
            candidates: Counterfactual candidates to verify.
            measure_delta: Callable returning ΔD = D(clean) − D(candidate)
                for one candidate. Injected so the gate stays model-free;
                exceptions are recorded as MEASUREMENT_FAILED, never raised.

        Returns:
            A :class:`GateResult` with surviving candidates and the full
            discard trail.
        """
        cfg = self.config
        n_clean = len(clean_tokens)
        valid: List[CandidateCF] = []
        discarded: List[Discarded] = []

        for cand in candidates:
            structural = self._structural_check(cand, n_clean)
            if structural is not None:
                discarded.append(structural)
                continue

            try:
                delta = float(measure_delta(cand))
            except Exception as exc:  # recorded, never raised (Part 6 rule)
                discarded.append(Discarded(
                    cand, DiscardReason.MEASUREMENT_FAILED,
                    f"measurement raised: {exc}",
                ))
                continue

            if abs(delta) < cfg.noise_floor:
                discarded.append(Discarded(
                    cand, DiscardReason.NULL_EFFECT,
                    f"|ΔD|={abs(delta):.4g} below noise floor "
                    f"{cfg.noise_floor:.4g}; this counterfactual does not "
                    "move the decision and therefore measures nothing.",
                ))
                continue

            valid.append(cand)

        result = GateResult(valid=valid, discarded=discarded, config=cfg)
        logger.info(
            "CF gate: %d/%d candidates valid (%s)",
            len(valid), len(candidates),
            ", ".join(f"{k}={v}" for k, v in
                      result.discard_report()["discarded_by_reason"].items())
            or "no discards",
        )
        return result

    # ------------------------------------------------------------------
    def _structural_check(
        self, cand: CandidateCF, n_clean: int
    ) -> Optional[Discarded]:
        cfg = self.config
        if cand.tokens is None:
            return Discarded(
                cand, DiscardReason.ALIGNMENT,
                "candidate has no token sequence; cannot verify alignment.",
            )
        n = len(cand.tokens)
        if n_clean <= 0:
            return Discarded(
                cand, DiscardReason.TASK_DRIFT, "empty clean prompt.",
            )
        ratio = n / n_clean
        if ratio < cfg.min_len_ratio or ratio > cfg.max_len_ratio:
            return Discarded(
                cand, DiscardReason.TASK_DRIFT,
                f"length ratio {ratio:.2f} outside "
                f"[{cfg.min_len_ratio}, {cfg.max_len_ratio}]; candidate no "
                "longer resembles the task.",
            )
        if cfg.require_alignment and n != n_clean:
            return Discarded(
                cand, DiscardReason.ALIGNMENT,
                f"token length {n} != clean length {n_clean}; position-wise "
                "patching undefined (set require_alignment=False to use "
                "anchor-based patching).",
            )
        return None
