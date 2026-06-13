"""Tests for glassbox/distributional.py — bootstrap CIs + stratified mean."""
import pytest

from glassbox.distributional import bootstrap_ci, faithfulness_ci, stratified_mean


# ── bootstrap_ci ────────────────────────────────────────────────────────────
def test_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_constant_values_tight_ci():
    ci = bootstrap_ci([0.6, 0.6, 0.6, 0.6])
    assert ci["mean"] == pytest.approx(0.6)
    assert ci["ci_low"] == pytest.approx(0.6)
    assert ci["ci_high"] == pytest.approx(0.6)


def test_single_value_degenerate():
    ci = bootstrap_ci([0.42])
    assert ci["n"] == 1
    assert ci["ci_low"] == ci["ci_high"] == ci["mean"] == pytest.approx(0.42)
    assert "degenerate" in ci["method"]


def test_spread_ci_brackets_mean():
    ci = bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0], seed=1)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["n"] == 5
    assert 0.0 <= ci["ci_low"] <= ci["ci_high"] <= 1.0


def test_ci_is_deterministic_given_seed():
    a = bootstrap_ci([0.1, 0.5, 0.9, 0.3], seed=7)
    b = bootstrap_ci([0.1, 0.5, 0.9, 0.3], seed=7)
    assert a == b


# ── faithfulness_ci ─────────────────────────────────────────────────────────
def test_faithfulness_ci_over_sample():
    per_prompt = [
        {"sufficiency": 0.7, "comprehensiveness": 0.4, "f1": 0.51},
        {"sufficiency": 0.8, "comprehensiveness": 0.3, "f1": 0.44},
        {"sufficiency": 0.6, "comprehensiveness": 0.5, "f1": 0.55},
        {"sufficiency": 0.75, "comprehensiveness": 0.35, "f1": 0.47},
    ]
    out = faithfulness_ci(per_prompt, seed=0)
    assert out["n_prompts"] == 4
    for key in ("sufficiency", "comprehensiveness", "f1"):
        assert out[key]["ci_low"] <= out[key]["mean"] <= out[key]["ci_high"]


def test_faithfulness_ci_drops_invalid_and_raises_when_empty():
    mixed = [{"error": "x"}, {"f1": 0.5, "sufficiency": 0.6}]
    out = faithfulness_ci(mixed)
    assert out["n_prompts"] == 1
    with pytest.raises(ValueError):
        faithfulness_ci([{"error": "no faithfulness"}])


# ── stratified_mean ─────────────────────────────────────────────────────────
def test_stratified_mean_weighted():
    means = {"credit": 0.6, "triage": 0.3}
    weights = {"credit": 100, "triage": 300}
    # (0.6*100 + 0.3*300) / 400 = (60 + 90)/400 = 0.375
    assert stratified_mean(means, weights) == pytest.approx(0.375)


def test_stratified_mean_zero_weight_raises():
    with pytest.raises(ValueError):
        stratified_mean({"a": 0.5}, {"a": 0})
