"""Tests for glassbox/validation.py — sample-size gate and held-out validation."""
import pytest

from glassbox.validation import (
    HeldOutValidationResult,
    HeldOutValidator,
    SampleSizeError,
    SampleSizeGate,
    SampleSizeWarning,
)


# ── SampleSizeGate ─────────────────────────────────────────────────────────
def test_gate_hard_block():
    gate = SampleSizeGate()
    with pytest.raises(SampleSizeError):
        gate.check(n=15)
    with pytest.raises(SampleSizeError):
        gate.check(n=5, context="batch")


def test_gate_soft_warn():
    gate = SampleSizeGate()
    with pytest.warns(SampleSizeWarning):
        gate.check(n=35)


def test_gate_passes_silently():
    gate = SampleSizeGate()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        gate.check(n=100)  # should not warn or raise


def test_gate_custom_thresholds():
    gate = SampleSizeGate(hard_minimum=5, soft_minimum=10)
    with pytest.warns(SampleSizeWarning):
        gate.check(n=7)  # between custom thresholds -> warns, does not raise
    with pytest.raises(SampleSizeError):
        gate.check(n=4)


def test_recommend_n_is_reasonable():
    n = SampleSizeGate().recommend_n()
    assert isinstance(n, int)
    assert 80 < n < 250  # Fisher-Z power analysis at rho=0.25, power=0.80
    # tighter detectable effect -> larger n
    assert SampleSizeGate().recommend_n(rho_min=0.1) > n


# ── HeldOutValidator ───────────────────────────────────────────────────────
def _result(f1, suff=1.0, comp=0.2):
    return {"faithfulness": {"f1": f1, "sufficiency": suff, "comprehensiveness": comp}}


def test_validate_requires_four():
    v = HeldOutValidator()
    with pytest.raises(ValueError):
        v.validate([_result(0.6), _result(0.6), _result(0.6)])  # only 3


def test_validate_generalises():
    # all four similar -> small gap -> generalises
    results = [_result(0.60), _result(0.62), _result(0.61), _result(0.63)]
    out = HeldOutValidator().validate(results)
    assert isinstance(out, HeldOutValidationResult)
    assert out.n_train == 2 and out.n_test == 2
    assert out.generalisation_gap < 0.10
    assert out.generalises is True
    assert out.overfit is False


def test_validate_detects_overfit():
    # first half high, second half low (no shuffle) -> large gap
    results = [_result(0.95), _result(0.93), _result(0.45), _result(0.40)]
    out = HeldOutValidator().validate(results)
    assert out.generalisation_gap >= 0.10
    assert out.overfit is True
    assert out.generalises is False


def test_validate_filters_invalid_results():
    results = [
        _result(0.6), _result(0.6), _result(0.6), _result(0.6),
        {"error": "model failed"},          # no faithfulness -> filtered
        {"something_else": 1},              # also filtered
    ]
    out = HeldOutValidator().validate(results)
    assert out.n_train + out.n_test == 4  # only the 4 valid counted


def test_validate_with_seed_runs():
    results = [_result(0.6 + i * 0.01) for i in range(8)]
    out = HeldOutValidator(seed=42).validate(results)
    assert out.n_train + out.n_test == 8
    assert sorted(out.train_indices + out.test_indices) == list(range(8))


def test_result_to_dict_and_summary():
    results = [_result(0.60), _result(0.62), _result(0.61), _result(0.63)]
    out = HeldOutValidator().validate(results)
    d = out.to_dict()
    assert d["n_train"] == 2
    assert "generalisation_gap" in d
    assert "gap_threshold" in d
    s = out.summary_line()
    assert "HeldOut" in s
    assert "F1_train" in s
