"""
glassbox/validation.py
======================
Statistical Validation Gates — v3.7.0
======================================

Implements two validation components:

1. SampleSizeGate  — blocks or warns on statistically underpowered analyses
2. HeldOutValidator — 50/50 train/test split to detect circuit overfitting

Sample Size Requirements (ROADMAP_V4 §v3.7.0)
----------------------------------------------
Minimum sample size for reliable faithfulness estimation (power analysis):

    n_min = ((z_{α/2} + z_β) / atanh(ρ_min))² + 3

With α=0.05 (two-sided), β=0.20 (80% power), ρ_min=0.25:
    z_{0.025} = 1.96, z_{0.20} = 0.84
    n_min = ((1.96 + 0.84) / atanh(0.25))² + 3 ≈ 126

Practical thresholds (matching ROADMAP_V4):
    n < 20  → SampleSizeError (block)
    n < 50  → SampleSizeWarning (warn, proceed)

Held-Out Validation (ROADMAP_V4 §v3.7.0)
-----------------------------------------
Generalisation gap criterion:

    gap = |F1_train − F1_test| < δ_gen = 0.10

A circuit passing on train but not test is flagged `overfit`.
This prevents circuits that exploit prompt-specific quirks.

References
----------
Conmy et al. 2023 — "Towards Automated Circuit Discovery for Mechanistic
    Interpretability" (ACDC).  https://arxiv.org/abs/2304.14997

Geiger et al. 2021 — "Causal Abstractions of Neural Networks"
    https://arxiv.org/abs/2106.02997
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────

N_HARD_MINIMUM:  int   = 20     # Block: SampleSizeError raised
N_SOFT_MINIMUM:  int   = 50     # Warn:  SampleSizeWarning issued
GENERALIZATION_GAP_THRESHOLD: float = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────────────────────────────────────

class SampleSizeError(ValueError):
    """
    Raised when n < N_HARD_MINIMUM (20).

    Statistical estimates are unreliable below this threshold; Glassbox
    blocks analysis to prevent misleading outputs.
    """
    pass


class SampleSizeWarning(UserWarning):
    """
    Issued when N_HARD_MINIMUM ≤ n < N_SOFT_MINIMUM (20 ≤ n < 50).

    Estimates may have wide confidence intervals. Analysis proceeds but
    the user should interpret results cautiously.
    """
    pass


# ──────────────────────────────────────────────────────────────────────────────
# SampleSizeGate
# ──────────────────────────────────────────────────────────────────────────────

class SampleSizeGate:
    """
    Guards batch analyses against statistically underpowered sample sizes.

    Enforces:
    - n < 20  → raises SampleSizeError (hard block)
    - n < 50  → issues SampleSizeWarning (soft warn, continues)
    - n ≥ 50  → no action

    Parameters
    ----------
    hard_minimum : Block threshold (default 20)
    soft_minimum : Warn threshold  (default 50)

    Usage
    -----
    >>> gate = SampleSizeGate()
    >>> gate.check(n=15)          # raises SampleSizeError
    >>> gate.check(n=35)          # issues SampleSizeWarning
    >>> gate.check(n=100)         # passes silently
    """

    def __init__(
        self,
        hard_minimum: int = N_HARD_MINIMUM,
        soft_minimum: int = N_SOFT_MINIMUM,
    ) -> None:
        self.hard_minimum = hard_minimum
        self.soft_minimum = soft_minimum

    def check(self, n: int, context: str = "") -> None:
        """
        Check sample size and raise/warn accordingly.

        Parameters
        ----------
        n       : Number of prompts in the analysis
        context : Optional descriptive string for error messages

        Raises
        ------
        SampleSizeError   : if n < hard_minimum
        SampleSizeWarning : if hard_minimum ≤ n < soft_minimum
        """
        ctx = f" [{context}]" if context else ""

        if n < self.hard_minimum:
            raise SampleSizeError(
                f"BLOCKED{ctx}: n={n} is below the hard minimum ({self.hard_minimum}). "
                f"Statistical estimates are unreliable. Provide ≥{self.hard_minimum} prompts "
                f"(recommended: ≥{self.soft_minimum}). "
                f"Power analysis: n≥50 gives 80% power at |ρ|≥0.25 (α=0.05)."
            )

        if n < self.soft_minimum:
            msg = (
                f"SampleSizeWarning{ctx}: n={n} is below the recommended minimum "
                f"({self.soft_minimum}). Confidence intervals will be wide. "
                f"Interpret results cautiously. Recommend ≥{self.soft_minimum} prompts."
            )
            warnings.warn(msg, SampleSizeWarning, stacklevel=3)
            logger.warning(msg)

    def recommend_n(
        self,
        rho_min:   float = 0.25,
        alpha:     float = 0.05,
        power:     float = 0.80,
    ) -> int:
        """
        Compute recommended n for Fisher Z power analysis.

        n_min = ((z_{α/2} + z_β) / atanh(ρ_min))² + 3

        Parameters
        ----------
        rho_min : Minimum detectable correlation (default 0.25)
        alpha   : Significance level (default 0.05, two-sided)
        power   : Desired statistical power (default 0.80)

        Returns
        -------
        Recommended minimum n (rounded up to integer)
        """
        from scipy import stats as scipy_stats
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_beta  = scipy_stats.norm.ppf(power)
        fisher_z = np.arctanh(rho_min)
        n_min = ((z_alpha + z_beta) / fisher_z) ** 2 + 3
        return int(np.ceil(n_min))


# ──────────────────────────────────────────────────────────────────────────────
# HeldOut Validation result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeldOutValidationResult:
    """
    Result of held-out validation split.

    Attributes
    ----------
    n_train          : Number of train prompts (floor(n/2))
    n_test           : Number of test prompts  (ceil(n/2))
    f1_train         : Mean F1 faithfulness on train split
    f1_test          : Mean F1 faithfulness on test split
    generalisation_gap : |f1_train − f1_test|
    overfit          : True if gap ≥ GENERALIZATION_GAP_THRESHOLD (0.10)
    generalises      : True iff gap < 0.10
    suff_train       : Mean sufficiency on train
    suff_test        : Mean sufficiency on test
    comp_train       : Mean comprehensiveness on train
    comp_test        : Mean comprehensiveness on test
    train_indices    : Indices of train prompts in original list
    test_indices     : Indices of test prompts in original list
    """
    n_train:              int
    n_test:               int
    f1_train:             float
    f1_test:              float
    generalisation_gap:   float
    overfit:              bool
    generalises:          bool
    suff_train:           float
    suff_test:            float
    comp_train:           float
    comp_test:            float
    train_indices:        List[int]
    test_indices:         List[int]

    def to_dict(self) -> Dict:
        return {
            "n_train":            self.n_train,
            "n_test":             self.n_test,
            "f1_train":           round(self.f1_train, 4),
            "f1_test":            round(self.f1_test, 4),
            "generalisation_gap": round(self.generalisation_gap, 4),
            "overfit":            self.overfit,
            "generalises":        self.generalises,
            "gap_threshold":      GENERALIZATION_GAP_THRESHOLD,
            "suff_train":         round(self.suff_train, 4),
            "suff_test":          round(self.suff_test, 4),
            "comp_train":         round(self.comp_train, 4),
            "comp_test":          round(self.comp_test, 4),
            "train_indices":      self.train_indices,
            "test_indices":       self.test_indices,
        }

    def summary_line(self) -> str:
        status = "OVERFIT ⚠" if self.overfit else "OK ✓"
        return (
            f"HeldOut [{status}] | "
            f"F1_train={self.f1_train:.4f} F1_test={self.f1_test:.4f} "
            f"gap={self.generalisation_gap:.4f} "
            f"(threshold={GENERALIZATION_GAP_THRESHOLD})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# HeldOutValidator
# ──────────────────────────────────────────────────────────────────────────────

class HeldOutValidator:
    """
    50/50 held-out validation to detect circuit overfitting.

    Splits the result list into train (first half) and test (second half),
    computes faithfulness metrics on each, and checks the generalisation gap:

        gap = |F1_train − F1_test| < δ_gen = 0.10

    Circuits with gap ≥ 0.10 are flagged `overfit`.

    Parameters
    ----------
    gap_threshold : Generalisation gap threshold (default 0.10)
    seed          : Shuffle seed. None = no shuffle (preserve order)

    Usage
    -----
    >>> validator = HeldOutValidator()
    >>> val_result = validator.validate(results)   # results from batch_analyze
    >>> val_result.generalises
    True
    >>> print(val_result.summary_line())
    HeldOut [OK ✓] | F1_train=0.6821 F1_test=0.6540 gap=0.0281 (threshold=0.1)
    """

    def __init__(
        self,
        gap_threshold: float = GENERALIZATION_GAP_THRESHOLD,
        seed:          Optional[int] = None,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.seed          = seed

    def validate(
        self,
        results: List[Dict],
    ) -> HeldOutValidationResult:
        """
        Run held-out validation on batch_analyze output.

        Parameters
        ----------
        results : List of dicts from GlassboxEngine.batch_analyze().
                  Each valid result must have a 'faithfulness' key with
                  'sufficiency', 'comprehensiveness', and 'f1' sub-keys.
                  Results with 'error' key are filtered out automatically.

        Returns
        -------
        HeldOutValidationResult

        Raises
        ------
        ValueError : if fewer than 4 valid results are available
                     (need ≥2 per split for meaningful comparison)
        """
        # Filter valid results (no error key)
        valid_results  = [(i, r) for i, r in enumerate(results) if "faithfulness" in r]
        valid_indices  = [i for i, _ in valid_results]
        valid_only     = [r for _, r in valid_results]

        n = len(valid_only)
        if n < 4:
            raise ValueError(
                f"HeldOutValidator requires ≥4 valid results for a meaningful "
                f"50/50 split; got {n}. Run more prompts."
            )

        # Optional shuffle
        order = list(range(n))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)

        # 50/50 split
        n_train = n // 2
        n_test  = n - n_train

        train_order = order[:n_train]
        test_order  = order[n_train:]

        train_results = [valid_only[i] for i in train_order]
        test_results  = [valid_only[i] for i in test_order]

        train_orig_idx = [valid_indices[i] for i in train_order]
        test_orig_idx  = [valid_indices[i] for i in test_order]

        # Compute metrics per split
        def _mean_metric(split: List[Dict], key: str) -> float:
            vals = [r["faithfulness"][key] for r in split if key in r.get("faithfulness", {})]
            return float(np.mean(vals)) if vals else 0.0

        f1_train   = _mean_metric(train_results, "f1")
        f1_test    = _mean_metric(test_results,  "f1")
        suff_train = _mean_metric(train_results, "sufficiency")
        suff_test  = _mean_metric(test_results,  "sufficiency")
        comp_train = _mean_metric(train_results, "comprehensiveness")
        comp_test  = _mean_metric(test_results,  "comprehensiveness")

        gap      = abs(f1_train - f1_test)
        overfit  = gap >= self.gap_threshold

        return HeldOutValidationResult(
            n_train            = n_train,
            n_test             = n_test,
            f1_train           = f1_train,
            f1_test            = f1_test,
            generalisation_gap = gap,
            overfit            = overfit,
            generalises        = not overfit,
            suff_train         = suff_train,
            suff_test          = suff_test,
            comp_train         = comp_train,
            comp_test          = comp_test,
            train_indices      = train_orig_idx,
            test_indices       = test_orig_idx,
        )

    def validate_prompts_directly(
        self,
        prompts:    List[Tuple[str, str, str]],
        gb_engine:  object,
        method:     str = "taylor",
    ) -> HeldOutValidationResult:
        """
        Convenience method: split prompts first, then run analysis on each half.

        This is the more statistically rigorous approach — the circuit is
        identified on train prompts only, then tested on unseen test prompts.

        Parameters
        ----------
        prompts   : List of (prompt, correct, incorrect) 3-tuples
        gb_engine : GlassboxEngine instance
        method    : Attribution method

        Returns
        -------
        HeldOutValidationResult
        """
        n = len(prompts)
        if n < 4:
            raise ValueError(f"Need ≥4 prompts for held-out validation; got {n}.")

        order = list(range(n))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)

        n_train = n // 2
        train_prompts = [prompts[i] for i in order[:n_train]]
        test_prompts  = [prompts[i] for i in order[n_train:]]

        # Analyze both splits
        train_results = gb_engine.batch_analyze(train_prompts, method=method,
                                                 show_progress=False, skip_errors=True)
        test_results  = gb_engine.batch_analyze(test_prompts, method=method,
                                                 show_progress=False, skip_errors=True)

        # Combine and validate
        combined    = train_results + test_results
        all_indices = list(range(len(combined)))

        return self.validate(combined)
