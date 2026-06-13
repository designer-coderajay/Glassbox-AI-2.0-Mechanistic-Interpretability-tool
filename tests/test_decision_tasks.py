"""Tests for the V5 decision-functional benchmark (dataset + scoring harness)."""
import pytest

from benchmarks.decision_tasks import DECISION_TASKS, DecisionTask, task_functional
from benchmarks.run_decision_functional import (
    assess_tier,
    attribution_concentration,
    build_report,
    format_table,
    run_task,
)


# ── dataset ─────────────────────────────────────────────────────────────────
def _fake_encoder():
    table = {}

    def enc(s):
        if s not in table:
            table[s] = len(table) + 1
        return [table[s]]

    return enc


def test_suite_nonempty_and_well_formed():
    assert len(DECISION_TASKS) >= 5
    names = [t.name for t in DECISION_TASKS]
    assert len(names) == len(set(names))  # unique
    for t in DECISION_TASKS:
        assert t.prompt.strip()
        assert t.expected in ("positive", "negative")
        assert t.positive_variants and t.negative_variants


def test_every_task_resolves_without_overlap():
    enc = _fake_encoder()
    for t in DECISION_TASKS:
        resolved = task_functional(t).resolve(enc)
        assert resolved.positive_ids and resolved.negative_ids
        pos = {tuple(i) for i in resolved.positive_ids}
        neg = {tuple(i) for i in resolved.negative_ids}
        assert not (pos & neg), f"{t.name}: positive/negative token overlap"


def test_bad_expected_rejected():
    with pytest.raises(ValueError):
        DecisionTask(
            name="x", domain="d", prompt="p",
            positive_label="a", positive_variants=(" A",),
            negative_label="b", negative_variants=(" B",),
            expected="maybe", annex_iii_ref="-",
        )


# ── attribution concentration (ERASER-fix random comparator) ────────────────
def test_concentration_flags_strong_circuit():
    attrs = {f"({l}, {h})": (0.5 if (l * 12 + h) < 14 else 0.001)
             for l in range(12) for h in range(12)}
    c = attribution_concentration(attrs, n_circuit=14)
    assert c["above_random"] is True
    assert c["concentration_ratio"] > 3.0
    assert c["n_total"] == 144


def test_concentration_uniform_is_not_above_random():
    attrs = {f"({l}, {h})": 0.1 for l in range(12) for h in range(12)}
    c = attribution_concentration(attrs, n_circuit=14)
    # uniform mass: circuit fraction == random expectation, so NOT above random
    assert c["above_random"] is False
    assert c["concentration_ratio"] == pytest.approx(1.0, abs=0.01)


def test_concentration_empty():
    c = attribution_concentration({}, n_circuit=14)
    assert c["above_random"] is False
    assert c["n_total"] == 0


# ── evidence tier ───────────────────────────────────────────────────────────
def test_single_prompt_tier_is_behavioral():
    t = assess_tier(has_weights=True, counterfactual_valid=True,
                    hessian_reliable=None, sample_n=1)
    assert t["tier"] == "C"  # no exact-patch/abstraction + hessian uncomputed
    assert any("underpowered" in r.lower() or "sample size" in r.lower()
               for r in t["reasons"])


def test_no_weights_tier_drops_to_behavioral_or_lower():
    t = assess_tier(has_weights=False, counterfactual_valid=None,
                    hessian_reliable=None, sample_n=None)
    assert t["tier"] in ("C", "D")


# ── run_task against a stub engine (no torch) ───────────────────────────────
class _StubEngine:
    def __init__(self, clean_ld, faith, attrs, circuit):
        self._clean_ld, self._faith, self._attrs, self._circuit = clean_ld, faith, attrs, circuit

    def analyze(self, prompt, correct, incorrect, method="taylor"):
        assert isinstance(correct, list) and isinstance(incorrect, list)
        return {
            "clean_ld": self._clean_ld,
            "faithfulness": self._faith,
            "circuit": self._circuit,
            "n_heads": len(self._circuit),
            "attributions": self._attrs,
        }


def _canned_attrs(n_strong=14):
    return {f"({l}, {h})": (0.5 if (l * 12 + h) < n_strong else 0.001)
            for l in range(12) for h in range(12)}


def test_run_task_model_correct():
    task = next(t for t in DECISION_TASKS if t.expected == "positive")
    eng = _StubEngine(
        clean_ld=2.5,
        faith={"sufficiency": 0.80, "comprehensiveness": 0.40, "f1": 0.53, "category": "moderate"},
        attrs=_canned_attrs(), circuit=[(0, 0)] * 14,
    )
    row = run_task(eng, task)
    assert row["matches_expected"] is True
    assert row["model_decision"] == "positive"
    assert row["f1"] == 0.53
    assert row["concentration"]["above_random"] is True
    assert row["tier"] == "C"


def test_run_task_model_wrong_is_recorded_not_hidden():
    task = next(t for t in DECISION_TASKS if t.expected == "positive")
    eng = _StubEngine(
        clean_ld=-1.8,  # model picks the negative outcome
        faith={"sufficiency": 0.3, "comprehensiveness": 0.2, "f1": 0.24, "category": "weak"},
        attrs=_canned_attrs(), circuit=[(0, 0)] * 14,
    )
    row = run_task(eng, task)
    assert row["matches_expected"] is False
    assert row["model_decision"] == "negative"


# ── report aggregation ──────────────────────────────────────────────────────
def test_run_task_skips_when_no_single_token_variant():
    task = DECISION_TASKS[0]
    eng = _StubEngine(
        clean_ld=2.0, faith={"sufficiency": 0.7, "comprehensiveness": 0.5, "f1": 0.58},
        attrs=_canned_attrs(), circuit=[(0, 0)] * 14,
    )
    row = run_task(eng, task, single_token=lambda s: False)  # nothing single-token
    assert row.get("skipped")
    assert row["matches_expected"] is False
    assert row["f1"] is None
    assert row["concentration"]["above_random"] is False


def test_run_task_filters_to_single_token_survivors():
    task = DECISION_TASKS[0]
    eng = _StubEngine(
        clean_ld=2.0, faith={"sufficiency": 0.7, "comprehensiveness": 0.5, "f1": 0.58},
        attrs=_canned_attrs(), circuit=[(0, 0)] * 14,
    )
    # only the capitalized forms survive; task should still run
    row = run_task(eng, task, single_token=lambda s: s in (" Yes", " No"))
    assert not row.get("skipped")
    assert row["f1"] == 0.58


def test_report_counts_skipped_and_table_renders():
    eng = _StubEngine(
        clean_ld=2.0, faith={"sufficiency": 0.7, "comprehensiveness": 0.5, "f1": 0.58},
        attrs=_canned_attrs(), circuit=[(0, 0)] * 14,
    )
    rows = [run_task(eng, DECISION_TASKS[0], single_token=lambda s: False)]
    rows += [run_task(eng, t) for t in DECISION_TASKS[1:]]
    report = build_report(rows, model="gpt2", method="taylor")
    assert report["n_skipped"] == 1
    assert format_table(report)  # does not raise on a skipped row


def test_build_report_and_table():
    eng = _StubEngine(
        clean_ld=2.0,
        faith={"sufficiency": 0.7, "comprehensiveness": 0.5, "f1": 0.58, "category": "moderate"},
        attrs=_canned_attrs(), circuit=[(0, 0)] * 14,
    )
    rows = [run_task(eng, t) for t in DECISION_TASKS]
    report = build_report(rows, model="gpt2", method="taylor")
    assert report["n_tasks"] == len(DECISION_TASKS)
    assert report["n_above_random"] == len(DECISION_TASKS)
    assert 0 <= report["n_model_correct"] <= len(DECISION_TASKS)
    table = format_table(report)
    assert "decision-functional benchmark" in table.lower()
    assert "model-correct" in table.lower()
