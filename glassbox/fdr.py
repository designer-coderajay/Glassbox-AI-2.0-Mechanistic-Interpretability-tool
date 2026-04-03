"""
glassbox/fdr.py
===============
Benjamini-Hochberg False Discovery Rate Control — v4.0.0
=========================================================

Implements multiple testing correction for head-level attribution significance,
adding Benjamini-Hochberg (BH) FDR control alongside the existing Bonferroni
correction.

Background: The Multiple Testing Problem
-----------------------------------------
When testing K attention heads simultaneously for significance, the probability
of at least one false positive under H₀ inflates rapidly. For K=144 heads
(GPT-2-small: 12 layers × 12 heads), a naive α=0.05 test produces ~7 false
positives in expectation.

Two correction approaches:

1. Bonferroni (FWER): α_adj = α/K
   - Controls Family-Wise Error Rate (P[≥1 false positive])
   - Very conservative: high false-negative rate for large K
   - Already reported in Glassbox output

2. Benjamini-Hochberg (FDR): E[FDR] ≤ α
   - Controls *expected fraction* of false positives among rejections
   - More powerful than Bonferroni for large K
   - Standard in neuroimaging (Genovese et al. 2002), used by Anthropic (2023)

BH Procedure (Benjamini & Hochberg, 1995)
------------------------------------------
Given p-values p_{(1)} ≤ p_{(2)} ≤ … ≤ p_{(K)}:

1. For each i = 1, …, K compute threshold: t_i = (i/K) · α
2. Find largest i where p_{(i)} ≤ t_i → call this i*
3. Reject all H₀_{(j)} for j ≤ i*

Mathematically: E[FDR] = (m₀/K) · α ≤ α, where m₀ = true null count.

Attribution p-values
--------------------
For head h with attribution α(h) and n independent bootstrap samples:

    z_h = α(h) / SE(α(h))     ~ N(0,1) under H₀: α(h) = 0
    p_h = 2 · Φ(-|z_h|)       (two-sided)

SE estimated via bootstrap (Efron & Tibshirani 1993) or Δ-method.

References
----------
Benjamini & Hochberg 1995 — "Controlling the False Discovery Rate"
    JRSS-B 57(1):289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Genovese et al. 2002 — "Thresholding of Statistical Maps in Functional
    Neuroimaging Using the False Discovery Rate"
    NeuroImage 15(4):870-878.

Anthropic Interpretability Team 2023 — "Towards Monosemanticity"
    Uses BH FDR for feature significance in sparse autoencoders.
    https://transformer-circuits.pub/2023/monosemanticity/index.html

scipy.stats.false_discovery_control — Reference implementation (scipy ≥1.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadSignificance:
    """
    Significance result for a single attention head.

    Attributes
    ----------
    layer           : Transformer layer index
    head            : Attention head index
    attribution     : Raw attribution score α(h)
    p_value         : Two-sided p-value under H₀: α(h) = 0
    p_bonferroni    : Bonferroni-corrected p-value (p × K)
    p_bh_adjusted   : BH-adjusted p-value (Benjamini & Hochberg 1995)
    significant_bh  : True if head survives BH FDR correction (q ≤ α)
    significant_bon : True if head survives Bonferroni correction
    bh_rank         : Rank of p-value in BH procedure (1 = most significant)
    """
    layer:           int
    head:            int
    attribution:     float
    p_value:         float
    p_bonferroni:    float
    p_bh_adjusted:   float
    significant_bh:  bool
    significant_bon: bool
    bh_rank:         int

    def to_dict(self) -> Dict:
        return {
            "head":            f"L{self.layer}H{self.head}",
            "layer":           self.layer,
            "head_idx":        self.head,
            "attribution":     round(self.attribution, 4),
            "p_value":         round(self.p_value, 6),
            "p_bonferroni":    round(self.p_bonferroni, 6),
            "p_bh_adjusted":   round(self.p_bh_adjusted, 6),
            "significant_bh":  self.significant_bh,
            "significant_bon": self.significant_bon,
            "bh_rank":         self.bh_rank,
        }


@dataclass
class FDRReport:
    """
    Full FDR report for all attention heads.

    Attributes
    ----------
    alpha                 : Nominal significance level (default 0.05)
    n_heads               : Total number of heads tested (K)
    n_significant_bh      : Number of heads significant after BH FDR
    n_significant_bonf    : Number of heads significant after Bonferroni
    expected_fdr          : Estimated E[FDR] = (n_null / K) × α
    bonferroni_threshold  : α / K
    bh_threshold          : BH critical threshold for the most significant head
    head_results          : List of HeadSignificance, sorted by p-value
    method                : "benjamini_hochberg" (always)
    """
    alpha:                float
    n_heads:              int
    n_significant_bh:     int
    n_significant_bonf:   int
    expected_fdr:         float
    bonferroni_threshold: float
    bh_threshold:         float
    head_results:         List[HeadSignificance]
    method:               str = "benjamini_hochberg"

    def to_dict(self) -> Dict:
        return {
            "method":                self.method,
            "alpha":                 self.alpha,
            "n_heads_tested":        self.n_heads,
            "n_significant_bh":      self.n_significant_bh,
            "n_significant_bonf":    self.n_significant_bonf,
            "expected_fdr":          round(self.expected_fdr, 4),
            "bonferroni_threshold":  round(self.bonferroni_threshold, 6),
            "bh_threshold":          round(self.bh_threshold, 6),
            "head_results":          [h.to_dict() for h in self.head_results],
        }

    def significant_heads_bh(self) -> List[Tuple[int, int]]:
        """Return (layer, head) tuples of BH-significant heads."""
        return [(h.layer, h.head) for h in self.head_results if h.significant_bh]

    def significant_heads_bonf(self) -> List[Tuple[int, int]]:
        """Return (layer, head) tuples of Bonferroni-significant heads."""
        return [(h.layer, h.head) for h in self.head_results if h.significant_bon]

    def summary_line(self) -> str:
        return (
            f"FDR [BH: {self.n_significant_bh}/{self.n_heads} significant | "
            f"Bonferroni: {self.n_significant_bonf}/{self.n_heads}] "
            f"E[FDR]≤{self.expected_fdr:.3f} α={self.alpha}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# P-value estimation utilities
# ──────────────────────────────────────────────────────────────────────────────

def attribution_to_pvalue(
    attribution:  float,
    se:           float,
) -> float:
    """
    Two-sided p-value for H₀: α(h) = 0 via z-test.

    z = α(h) / SE(α(h))
    p = 2 · Φ(-|z|)

    Parameters
    ----------
    attribution : Head attribution score α(h)
    se          : Standard error of the attribution (from bootstrap or Δ-method)

    Returns
    -------
    p-value ∈ (0, 1]; returns 1.0 if SE=0 (degenerate)
    """
    if se <= 0 or se != se:  # zero or NaN
        return 1.0
    z = attribution / se
    return float(2 * scipy_stats.norm.sf(abs(z)))


def bootstrap_se(
    attributions_per_bootstrap: List[Dict[Tuple[int, int], float]],
    head:                        Tuple[int, int],
) -> float:
    """
    Compute bootstrap SE for a single head attribution.

    SE = std(α_boot(h))  over B bootstrap samples.

    Parameters
    ----------
    attributions_per_bootstrap : List of attribution dicts, one per bootstrap replicate
    head                       : (layer, head_idx) tuple

    Returns
    -------
    Bootstrap standard error (float)
    """
    vals = [b.get(head, 0.0) for b in attributions_per_bootstrap]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# BH FDR Controller
# ──────────────────────────────────────────────────────────────────────────────

class BenjaminiHochberg:
    """
    Benjamini-Hochberg FDR control for head-level attribution significance.

    Controls E[FDR] ≤ α over all K head tests simultaneously. Reports both
    BH and Bonferroni thresholds for comparison.

    Parameters
    ----------
    alpha : Nominal significance level (default 0.05)

    Usage
    -----
    Basic (single run with fixed SE):

    >>> bh = BenjaminiHochberg(alpha=0.05)
    >>> attributions = {(9, 6): 0.584, (9, 9): 0.431, ...}  # all K heads
    >>> se_map       = {(9, 6): 0.112, (9, 9): 0.098, ...}  # bootstrap SE
    >>> report = bh.run(attributions, se_map)
    >>> print(report.summary_line())
    FDR [BH: 8/144 significant | Bonferroni: 5/144] E[FDR]≤0.045 α=0.05

    Bootstrap mode (recommended for small n):

    >>> report = bh.run_bootstrap(
    ...     attributions_per_sample=bootstrap_attr_list,
    ...     observed_attributions=attributions,
    ... )
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha

    def run(
        self,
        attributions: Dict[Tuple[int, int], float],
        se_map:       Dict[Tuple[int, int], float],
    ) -> FDRReport:
        """
        Run BH FDR + Bonferroni on head attributions with provided SEs.

        Parameters
        ----------
        attributions : {(layer, head): attr_score} for ALL heads
        se_map       : {(layer, head): standard_error} — must cover all keys in attributions

        Returns
        -------
        FDRReport with per-head significance results
        """
        heads = sorted(attributions.keys())
        K     = len(heads)

        if K == 0:
            raise ValueError("attributions dict is empty")

        # Compute p-values
        pvals = np.array([
            attribution_to_pvalue(attributions[h], se_map.get(h, 1.0))
            for h in heads
        ])

        return self._compute_report(heads, pvals, attributions, K)

    def run_bootstrap(
        self,
        attributions_per_sample: List[Dict[Tuple[int, int], float]],
        observed_attributions:   Dict[Tuple[int, int], float],
    ) -> FDRReport:
        """
        Run BH FDR using bootstrap SE estimates.

        Parameters
        ----------
        attributions_per_sample : List of attribution dicts, one per bootstrap replicate
        observed_attributions   : Attribution dict from the actual (non-bootstrapped) run

        Returns
        -------
        FDRReport
        """
        heads = sorted(observed_attributions.keys())
        K     = len(heads)

        se_map: Dict[Tuple[int, int], float] = {}
        for h in heads:
            se_map[h] = bootstrap_se(attributions_per_sample, h)

        return self.run(observed_attributions, se_map)

    def run_permutation(
        self,
        attributions_per_permutation: List[Dict[Tuple[int, int], float]],
        observed_attributions:        Dict[Tuple[int, int], float],
    ) -> FDRReport:
        """
        Run BH FDR using permutation-based p-values.

        p_h = fraction of permutation runs where |α_perm(h)| ≥ |α_obs(h)|

        Parameters
        ----------
        attributions_per_permutation : List of attribution dicts from permuted runs
        observed_attributions        : Observed attribution dict

        Returns
        -------
        FDRReport
        """
        heads = sorted(observed_attributions.keys())
        K     = len(heads)
        B     = len(attributions_per_permutation)

        pvals = np.zeros(K)
        for i, h in enumerate(heads):
            obs = abs(observed_attributions[h])
            perm_vals = [abs(b.get(h, 0.0)) for b in attributions_per_permutation]
            pvals[i]  = (sum(v >= obs for v in perm_vals) + 1) / (B + 1)

        return self._compute_report(heads, pvals, observed_attributions, K)

    def _compute_report(
        self,
        heads:        List[Tuple[int, int]],
        pvals:        np.ndarray,
        attributions: Dict[Tuple[int, int], float],
        K:            int,
    ) -> FDRReport:
        """Internal: apply BH and Bonferroni, build FDRReport."""

        # Sort indices by p-value (ascending)
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]

        # BH thresholds: t_i = (i+1)/K * alpha  (1-indexed i)
        bh_thresholds = (np.arange(1, K + 1) / K) * self.alpha

        # Find largest i where p_{(i)} ≤ t_i
        reject_mask = sorted_pvals <= bh_thresholds
        if reject_mask.any():
            i_star = int(np.where(reject_mask)[0].max())
        else:
            i_star = -1

        # BH-adjusted p-values (step-up method)
        # p_adj_{(i)} = min_{j≥i} (K/j) * p_{(j)}
        bh_adjusted = np.minimum.accumulate(
            (K / np.arange(1, K + 1))[::-1] * sorted_pvals[::-1]
        )[::-1]
        bh_adjusted = np.minimum(bh_adjusted, 1.0)

        # Map back to head order
        bh_adj_per_head = np.empty(K)
        bh_adj_per_head[sorted_idx] = bh_adjusted

        significant_bh_set  = set(sorted_idx[:i_star + 1]) if i_star >= 0 else set()
        bonferroni_threshold = self.alpha / K

        head_results: List[HeadSignificance] = []
        for i, h in enumerate(heads):
            rank = int(np.where(sorted_idx == i)[0][0]) + 1   # 1-indexed
            head_results.append(HeadSignificance(
                layer           = h[0],
                head            = h[1],
                attribution     = attributions[h],
                p_value         = float(pvals[i]),
                p_bonferroni    = float(min(pvals[i] * K, 1.0)),
                p_bh_adjusted   = float(bh_adj_per_head[i]),
                significant_bh  = i in {sorted_idx[j] for j in range(i_star + 1)} if i_star >= 0 else False,
                significant_bon = pvals[i] <= bonferroni_threshold,
                bh_rank         = rank,
            ))

        # Sort results by p-value
        head_results.sort(key=lambda x: x.p_value)

        n_sig_bh   = sum(1 for h in head_results if h.significant_bh)
        n_sig_bonf = sum(1 for h in head_results if h.significant_bon)

        # Estimate E[FDR] = (n_null / K) × α, n_null ≈ K − n_sig_bh
        n_null      = max(0, K - n_sig_bh)
        expected_fdr = (n_null / K) * self.alpha if K > 0 else 0.0

        bh_thresh = float(bh_thresholds[i_star]) if i_star >= 0 else 0.0

        return FDRReport(
            alpha                 = self.alpha,
            n_heads               = K,
            n_significant_bh      = n_sig_bh,
            n_significant_bonf    = n_sig_bonf,
            expected_fdr          = expected_fdr,
            bonferroni_threshold  = bonferroni_threshold,
            bh_threshold          = bh_thresh,
            head_results          = head_results,
            method                = "benjamini_hochberg",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def apply_fdr_correction(
    attributions: Dict[Tuple[int, int], float],
    se_map:       Dict[Tuple[int, int], float],
    alpha:        float = 0.05,
) -> FDRReport:
    """
    Convenience wrapper: apply BH FDR to head attributions.

    Parameters
    ----------
    attributions : {(layer, head): attr_score} for ALL heads
    se_map       : {(layer, head): standard_error}
    alpha        : Significance level (default 0.05)

    Returns
    -------
    FDRReport
    """
    return BenjaminiHochberg(alpha=alpha).run(attributions, se_map)
