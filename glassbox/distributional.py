# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.distributional
=======================
V5 distributional faithfulness with confidence bounds
(ROADMAP_V5_FOUNDATIONS.md Part 2.3).

A single-prompt faithfulness number is not an audit. The deep-audit loop draws a
stratified sample from the production input distribution and reports
**population-level** faithfulness with confidence intervals — turning "F1 = 0.64
on one prompt" into "F1 = 0.61 [0.55, 0.67] over the credit-decision stratum."

Pure / numpy. The per-prompt scores come from the (torch) engine; everything here
operates on the resulting numbers and is unit-tested.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

__all__ = ["bootstrap_ci", "faithfulness_ci", "stratified_mean"]


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Args:
        values: Sample observations (e.g. per-prompt F1 scores).
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``{mean, ci_low, ci_high, n, alpha, method}``. For ``n == 1`` the CI is
        degenerate (low == high == mean) and labeled as such.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    mean = float(arr.mean())
    if n == 1:
        return {"mean": round(mean, 4), "ci_low": round(mean, 4),
                "ci_high": round(mean, 4), "n": 1, "alpha": alpha,
                "method": "degenerate (n=1)"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "n": n, "alpha": alpha, "method": "percentile_bootstrap"}


def faithfulness_ci(
    per_prompt: List[Dict[str, Any]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Population faithfulness CIs from a sample of per-prompt results.

    Args:
        per_prompt: List of result dicts, each with 'sufficiency',
            'comprehensiveness', and/or 'f1'. Records lacking 'f1' are dropped.

    Returns:
        Vault-ready dict: ``{n_prompts, alpha, sufficiency: ci, comprehensiveness:
        ci, f1: ci}`` — the §4 distributional evidence for the Annex IV file.
    """
    valid = [r for r in per_prompt if "f1" in r]
    if not valid:
        raise ValueError("no records with an 'f1' field")
    out: Dict[str, Any] = {"n_prompts": len(valid), "alpha": alpha}
    for key in ("sufficiency", "comprehensiveness", "f1"):
        vals = [float(r[key]) for r in valid if key in r and r[key] is not None]
        out[key] = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed) if vals else None
    return out


def stratified_mean(
    stratum_means: Dict[Any, float],
    stratum_weights: Dict[Any, float],
) -> float:
    """Population mean from per-stratum means and weights (e.g. stratum sizes).

    population_mean = Σ_s w_s · mean_s / Σ_s w_s, over strata present in both maps.
    """
    common = [s for s in stratum_means if s in stratum_weights]
    total_w = sum(stratum_weights[s] for s in common)
    if total_w <= 0:
        raise ValueError("total stratum weight must be > 0")
    return sum(stratum_means[s] * stratum_weights[s] for s in common) / total_w
