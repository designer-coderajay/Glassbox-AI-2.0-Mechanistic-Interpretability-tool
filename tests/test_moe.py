"""Tests for glassbox/moe.py — MoE partition + expert attribution + conformance proof."""
from contextlib import nullcontext

import pytest

from glassbox.auditable import UnitSpec, run_conformance
from glassbox.moe import expert_attribution, moe_units


# ── moe_units ───────────────────────────────────────────────────────────────
def test_moe_units_counts_and_kinds():
    units = moe_units(2, 4)
    assert len(units) == 2 * (1 + 4)  # router + 4 experts per layer
    assert sum(1 for u in units if u.kind == "router") == 2
    assert sum(1 for u in units if u.kind == "expert") == 8
    assert all(isinstance(u, UnitSpec) for u in units)
    # router precedes its experts within a layer
    assert units[0].name == "L0.router"
    assert units[1].name == "L0.expert0"


def test_moe_units_without_router():
    units = moe_units(3, 2, include_router=False)
    assert len(units) == 6
    assert all(u.kind == "expert" for u in units)


def test_moe_units_invalid():
    with pytest.raises(ValueError):
        moe_units(0, 4)
    with pytest.raises(ValueError):
        moe_units(2, 0)


# ── expert_attribution ──────────────────────────────────────────────────────
def test_expert_attribution_weight_times_contribution():
    routing = {"e0": 0.9, "e1": 0.1, "e2": 0.0}
    contrib = {"e0": 1.0, "e1": 2.0, "e2": 5.0}
    ranked = expert_attribution(routing, contrib)
    assert ranked[0] == ("e0", pytest.approx(0.9))      # 0.9*1.0
    # e2 has high contribution but zero routing weight -> zero causal share
    assert dict(ranked)["e2"] == pytest.approx(0.0)
    assert [e for e, _ in ranked] == ["e0", "e1", "e2"]  # sorted by |attr|


def test_expert_attribution_missing_weight_is_zero():
    ranked = expert_attribution({}, {"e0": 3.0})
    assert ranked == [("e0", 0.0)]


# ── conformance proof: the AuditableModel abstraction covers MoE ─────────────
class _MockMoE:
    """A deterministic mock MoE that implements the AuditableModel protocol."""

    config = {"n_layers": 2, "n_experts": 4}

    def forward(self, tokens):
        return [[0.2, 0.5, 0.3]]

    def units(self):
        return moe_units(2, 4)

    def read(self, unit, tokens):
        return [0.0]

    def patch(self, unit, value):
        return nullcontext()


def test_mock_moe_passes_conformance():
    report = run_conformance(_MockMoE(), tokens=[1, 2, 3])
    assert report.passed is True
    # determinism + patch_identity hold across all 10 expert/router units
    assert {c.name for c in report.checks} == {"determinism", "patch_identity"}
