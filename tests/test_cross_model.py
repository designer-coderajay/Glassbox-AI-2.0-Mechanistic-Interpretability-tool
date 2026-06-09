# SPDX-License-Identifier: MIT
"""
tests/test_cross_model.py — coverage for cross-model comparison logic.

The model-loading run() path needs real models, but the entire comparison core —
normalised circuits/attributions, Jaccard, attribution correlation, shared/
consensus heads, and the report serialisation — is pure computation over result
objects. All of that is tested here with constructed results, no model required.
"""

import pytest

from glassbox.cross_model import (
    CrossModelComparison,
    CrossModelReport,
    CrossModelSimilarity,
    ModelAnalysisConfig,
    SingleModelResult,
)


def _smr(name, circuit, attrs, n_layers=12, n_heads=12, ld=3.0, suff=0.9, comp=0.7):
    return SingleModelResult(
        model_name=name, n_layers=n_layers, n_heads=n_heads,
        circuit=circuit, attributions=attrs, clean_ld=ld,
        sufficiency=suff, comprehensiveness=comp,
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestConfig:
    def test_construction_defaults(self):
        c = ModelAnalysisConfig(
            model_name="gpt2", clean_prompt="p", corrupted_prompt="q",
            target_token=" Mary", distractor_token=" John")
        assert c.model_name == "gpt2" and c.device == "cpu"


class TestSingleModelResult:
    def test_normalised_circuit(self):
        r = _smr("m", [(0, 0), (6, 6)], {})
        assert r.normalised_circuit() == {(0.0, 0.0), (0.5, 0.5)}

    def test_normalised_attributions_scales_by_ld(self):
        r = _smr("m", [(0, 0)], {(0, 0): 3.0}, ld=3.0)
        assert r.normalised_attributions() == {(0.0, 0.0): 1.0}

    def test_normalised_attributions_zero_ld_uses_one(self):
        r = _smr("m", [(0, 0)], {(0, 0): 2.0}, ld=0.0)
        assert r.normalised_attributions() == {(0.0, 0.0): 2.0}


# ---------------------------------------------------------------------------
# Static similarity helpers
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical(self):
        s = {(0.0, 0.0), (0.5, 0.5)}
        assert CrossModelComparison._jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert CrossModelComparison._jaccard({(0.0, 0.0)}, {(0.9, 0.9)}) == 0.0

    def test_both_empty(self):
        assert CrossModelComparison._jaccard(set(), set()) == 0.0

    def test_partial(self):
        a = {(0.0, 0.0), (0.5, 0.5)}
        b = {(0.0, 0.0), (0.9, 0.9)}
        assert CrossModelComparison._jaccard(a, b) == pytest.approx(1 / 3)


class TestSharedHeads:
    def test_shared(self):
        a = {(0.0, 0.0), (0.5, 0.5)}
        b = {(0.0, 0.0), (0.9, 0.9)}
        shared = CrossModelComparison._shared_normalised_heads(a, b)
        assert (0.0, 0.0) in shared and len(shared) == 1


class TestAttributionPearson:
    def test_too_few_shared_returns_zero(self):
        a = _smr("a", [(0, 0)], {(0, 0): 1.0})
        b = _smr("b", [(8, 8)], {(8, 8): 1.0})
        assert CrossModelComparison._attribution_pearsonr(a, b) == 0.0

    def test_enough_shared_returns_correlation(self):
        attrs = {(0, 0): 1.0, (2, 2): 2.0, (4, 4): 3.0, (8, 8): 4.0}
        a = _smr("a", list(attrs), attrs)
        b = _smr("b", list(attrs), attrs)
        r = CrossModelComparison._attribution_pearsonr(a, b)
        assert isinstance(r, float) and -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Instance comparison methods (no model)
# ---------------------------------------------------------------------------

class TestComparisonMethods:
    def test_pairwise_similarities_count(self):
        cmp = CrossModelComparison([])
        results = [
            _smr("a", [(0, 0), (4, 4)], {(0, 0): 1.0, (4, 4): 2.0}),
            _smr("b", [(0, 0), (8, 8)], {(0, 0): 1.0, (8, 8): 2.0}),
            _smr("c", [(4, 4)], {(4, 4): 1.0}),
        ]
        sims = cmp._compute_pairwise_similarities(results)
        assert len(sims) == 3  # 3 choose 2
        assert all(isinstance(s, CrossModelSimilarity) for s in sims)

    def test_consensus_heads(self):
        cmp = CrossModelComparison([])
        results = [
            _smr("a", [(0, 0), (4, 4)], {}),
            _smr("b", [(0, 0), (8, 8)], {}),
        ]
        # (0,0) appears in both (100%) -> consensus at >=50%
        consensus = cmp._find_consensus_heads(results)
        assert (0.0, 0.0) in consensus

    def test_consensus_empty_results(self):
        assert CrossModelComparison([])._find_consensus_heads([]) == []


# ---------------------------------------------------------------------------
# CrossModelReport
# ---------------------------------------------------------------------------

@pytest.fixture
def report():
    results = [
        _smr("gpt2", [(0, 0), (4, 4)], {(0, 0): 1.0, (4, 4): 2.0}),
        _smr("pythia", [(0, 0), (8, 8)], {(0, 0): 1.5, (8, 8): 2.5}),
    ]
    cmp = CrossModelComparison([])
    sims = cmp._compute_pairwise_similarities(results)
    consensus = cmp._find_consensus_heads(results)
    return CrossModelReport(
        task_description="Indirect Object Identification",
        results=results, similarities=sims, consensus_heads=consensus,
    )


class TestReport:
    def test_summary(self, report):
        s = report.summary
        assert "Cross-Model Circuit Analysis" in s
        assert "gpt2" in s and "Jaccard" in s

    def test_summary_empty(self):
        r = CrossModelReport(task_description="t", results=[], similarities=[], consensus_heads=[])
        assert r.summary == "No results."

    def test_to_dict(self, report):
        d = report.to_dict()
        assert d["n_models"] == 2
        assert len(d["results"]) == 2 and len(d["similarities"]) == 1
        assert "consensus_heads" in d

    def test_attribution_table(self, report):
        t = report.attribution_table
        assert "| Model |" in t and "gpt2" in t and "pythia" in t

    def test_attribution_table_empty(self):
        r = CrossModelReport(task_description="t", results=[], similarities=[], consensus_heads=[])
        assert r.attribution_table == "No results to tabulate."
