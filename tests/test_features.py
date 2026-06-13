"""Tests for glassbox/features.py — feature partition + sparse attribution."""
from contextlib import nullcontext

import pytest

from glassbox.auditable import run_conformance
from glassbox.features import feature_units, sparse_feature_attribution


def test_feature_units_enumeration():
    units = feature_units({3: 2, 5: 3})
    assert len(units) == 5
    assert all(u.kind == "feature" for u in units)
    assert units[0].name == "L3.f0"
    assert {u.layer for u in units} == {3, 5}


def test_feature_units_invalid():
    with pytest.raises(ValueError):
        feature_units({0: -1})


def test_sparse_attribution_skips_inactive():
    activations = {"f0": 0.0, "f1": 0.8, "f2": 0.5}
    contributions = {"f0": 9.0, "f1": 1.0, "f2": 2.0}
    ranked = sparse_feature_attribution(activations, contributions)
    keys = [k for k, _ in ranked]
    assert "f0" not in keys                      # inactive -> no causal effect
    assert ranked[0][0] == "f2"                  # 0.5*2.0 = 1.0 > 0.8*1.0 = 0.8
    assert dict(ranked)["f1"] == pytest.approx(0.8)


def test_sparse_attribution_eps_threshold():
    ranked = sparse_feature_attribution(
        {"f0": 0.05, "f1": 0.9}, {"f0": 10.0, "f1": 1.0}, eps=0.1
    )
    assert [k for k, _ in ranked] == ["f1"]      # f0 below eps -> dropped


class _MockSAE:
    """Deterministic mock implementing AuditableModel with feature units."""

    def forward(self, tokens):
        return [[0.1, 0.9]]

    def units(self):
        return feature_units({0: 4, 1: 4})

    def read(self, unit, tokens):
        return [0.0]

    def patch(self, unit, value):
        return nullcontext()


def test_mock_sae_passes_conformance():
    report = run_conformance(_MockSAE(), tokens=[1, 2])
    assert report.passed is True
