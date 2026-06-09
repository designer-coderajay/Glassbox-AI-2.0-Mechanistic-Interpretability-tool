"""Tests for glassbox/fdr.py — Benjamini-Hochberg FDR control over head attributions."""
import math

import pytest

from glassbox.fdr import (
    BenjaminiHochberg,
    FDRReport,
    HeadSignificance,
    apply_fdr_correction,
    attribution_to_pvalue,
    bootstrap_se,
)


# ── attribution_to_pvalue ──────────────────────────────────────────────────
def test_pvalue_degenerate_se():
    # SE <= 0 or NaN is degenerate -> p = 1.0
    assert attribution_to_pvalue(0.5, 0.0) == 1.0
    assert attribution_to_pvalue(0.5, -0.1) == 1.0
    assert attribution_to_pvalue(0.5, float("nan")) == 1.0


def test_pvalue_zero_attribution():
    # z = 0 -> p = 2 * 0.5 = 1.0
    assert attribution_to_pvalue(0.0, 1.0) == pytest.approx(1.0)


def test_pvalue_large_z_is_tiny():
    p = attribution_to_pvalue(0.584, 0.05)  # z ~ 11.7
    assert 0.0 <= p < 1e-6


def test_pvalue_in_unit_interval():
    p = attribution_to_pvalue(0.1, 0.1)  # z = 1.0
    assert 0.0 < p <= 1.0
    assert p == pytest.approx(0.3173, abs=1e-3)


# ── bootstrap_se ───────────────────────────────────────────────────────────
def test_bootstrap_se_basic():
    samples = [{(9, 6): 0.5}, {(9, 6): 0.6}, {(9, 6): 0.55}]
    se = bootstrap_se(samples, (9, 6))
    assert se == pytest.approx(0.05, abs=1e-6)


def test_bootstrap_se_single_sample_is_zero():
    assert bootstrap_se([{(9, 6): 0.5}], (9, 6)) == 0.0


def test_bootstrap_se_missing_head_uses_zero():
    se = bootstrap_se([{}, {}, {}], (9, 6))
    assert se == 0.0


# ── BenjaminiHochberg construction ─────────────────────────────────────────
def test_bh_invalid_alpha():
    with pytest.raises(ValueError):
        BenjaminiHochberg(alpha=0.0)
    with pytest.raises(ValueError):
        BenjaminiHochberg(alpha=1.0)
    with pytest.raises(ValueError):
        BenjaminiHochberg(alpha=1.5)


def test_bh_empty_raises():
    with pytest.raises(ValueError):
        BenjaminiHochberg().run({}, {})


# ── BenjaminiHochberg.run ──────────────────────────────────────────────────
def _attr_se():
    attributions = {
        (9, 6): 0.584, (9, 9): 0.431, (10, 0): 0.312,
        (0, 0): 0.001, (0, 1): 0.002, (0, 2): 0.0015,
    }
    se_map = {h: 0.05 for h in attributions}
    return attributions, se_map


def test_bh_run_flags_strong_heads():
    attributions, se_map = _attr_se()
    report = BenjaminiHochberg(alpha=0.05).run(attributions, se_map)
    assert isinstance(report, FDRReport)
    assert report.n_heads == 6
    assert report.n_significant_bh >= 3
    sig = report.significant_heads_bh()
    assert (9, 6) in sig
    assert (0, 0) not in sig  # near-zero attribution is not significant


def test_bh_report_fields_and_helpers():
    attributions, se_map = _attr_se()
    report = apply_fdr_correction(attributions, se_map, alpha=0.05)
    d = report.to_dict()
    assert d["n_heads_tested"] == 6
    assert 0.0 <= report.expected_fdr <= 0.05
    assert report.bonferroni_threshold == pytest.approx(0.05 / 6)
    assert "FDR" in report.summary_line()
    # head_results sorted ascending by p-value
    pvals = [h.p_value for h in report.head_results]
    assert pvals == sorted(pvals)
    # Bonferroni set is a subset of (or equal to) BH set in size
    assert report.n_significant_bonf <= report.n_significant_bh


def test_head_significance_to_dict():
    attributions, se_map = _attr_se()
    report = BenjaminiHochberg().run(attributions, se_map)
    hs = report.head_results[0]
    assert isinstance(hs, HeadSignificance)
    hd = hs.to_dict()
    assert "p_value" in hd
    assert "layer" in hd or "head" in hd or len(hd) > 0


# ── bootstrap / permutation modes ──────────────────────────────────────────
def test_bh_run_bootstrap():
    observed = {(9, 6): 0.584, (0, 0): 0.001}
    samples = [
        {(9, 6): 0.58, (0, 0): 0.0},
        {(9, 6): 0.60, (0, 0): 0.002},
        {(9, 6): 0.57, (0, 0): 0.001},
        {(9, 6): 0.59, (0, 0): 0.0015},
    ]
    report = BenjaminiHochberg().run_bootstrap(samples, observed)
    assert report.n_heads == 2
    assert (9, 6) in report.significant_heads_bh()


def test_bh_run_permutation():
    observed = {(9, 6): 0.584, (0, 0): 0.001}
    perms = [{(9, 6): 0.01, (0, 0): 0.001} for _ in range(50)]
    report = BenjaminiHochberg().run_permutation(perms, observed)
    assert report.n_heads == 2
    # strong observed head should get a small permutation p-value
    strong = next(h for h in report.head_results if (h.layer, h.head) == (9, 6))
    assert strong.p_value < 0.1


def test_significant_heads_bonf_returns_list():
    attributions, se_map = _attr_se()
    report = BenjaminiHochberg().run(attributions, se_map)
    bonf = report.significant_heads_bonf()
    assert isinstance(bonf, list)
    for h in bonf:
        assert isinstance(h, tuple) and len(h) == 2
