"""
tests/test_cf_gate.py — V5 counterfactual verification gate (ROADMAP_V5 §3.3).

The contract: a counterfactual is used only if it (a) preserves the task
shape, (b) is alignable to the clean prompt, and (c) actually moves the
decision functional. Invalid candidates are DISCARDED AND REPORTED —
silence is how tools lie.

Pure logic: ΔD measurements are injected; no model required.
"""

from glassbox.cf_gate import (
    CandidateCF,
    CounterfactualGate,
    DiscardReason,
    GateConfig,
)

CLEAN = list(range(12))  # 12-token clean prompt stand-in


def cand(tokens, strategy="name_swap"):
    return CandidateCF(text="<cf>", strategy=strategy, tokens=tokens)


def measure_const(delta):
    def _measure(_c):
        return delta
    return _measure


class TestTaskShape:
    def test_same_length_passes(self):
        g = CounterfactualGate(GateConfig(noise_floor=0.1))
        res = g.evaluate(CLEAN, [cand(list(range(100, 112)))], measure_const(1.0))
        assert len(res.valid) == 1 and not res.discarded

    def test_grossly_shorter_candidate_discarded_as_task_drift(self):
        g = CounterfactualGate(GateConfig())
        res = g.evaluate(CLEAN, [cand([1, 2])], measure_const(1.0))
        assert not res.valid
        assert res.discarded[0].reason is DiscardReason.TASK_DRIFT

    def test_grossly_longer_candidate_discarded(self):
        g = CounterfactualGate(GateConfig())
        res = g.evaluate(CLEAN, [cand(list(range(40)))], measure_const(1.0))
        assert res.discarded[0].reason is DiscardReason.TASK_DRIFT


class TestAlignment:
    def test_unequal_length_discarded_when_alignment_required(self):
        g = CounterfactualGate(GateConfig(require_alignment=True))
        res = g.evaluate(CLEAN, [cand(list(range(11)))], measure_const(1.0))
        assert res.discarded[0].reason is DiscardReason.ALIGNMENT

    def test_unequal_length_allowed_when_alignment_not_required(self):
        g = CounterfactualGate(
            GateConfig(require_alignment=False, min_len_ratio=0.5)
        )
        res = g.evaluate(CLEAN, [cand(list(range(11)))], measure_const(1.0))
        assert len(res.valid) == 1


class TestEffectSize:
    def test_null_effect_discarded(self):
        """A counterfactual that changes nothing measures nothing."""
        g = CounterfactualGate(GateConfig(noise_floor=0.05))
        res = g.evaluate(CLEAN, [cand(list(range(100, 112)))],
                         measure_const(0.0))
        assert res.discarded[0].reason is DiscardReason.NULL_EFFECT

    def test_effect_exactly_at_floor_is_kept(self):
        g = CounterfactualGate(GateConfig(noise_floor=0.05))
        res = g.evaluate(CLEAN, [cand(list(range(100, 112)))],
                         measure_const(0.05))
        assert len(res.valid) == 1

    def test_negative_effects_count_by_magnitude(self):
        g = CounterfactualGate(GateConfig(noise_floor=0.05))
        res = g.evaluate(CLEAN, [cand(list(range(100, 112)))],
                         measure_const(-0.4))
        assert len(res.valid) == 1

    def test_measurement_exception_recorded_not_raised(self):
        g = CounterfactualGate(GateConfig())

        def boom(_c):
            raise RuntimeError("OOM")

        res = g.evaluate(CLEAN, [cand(list(range(100, 112)))], boom)
        assert res.discarded[0].reason is DiscardReason.MEASUREMENT_FAILED
        assert "OOM" in res.discarded[0].detail


class TestDiscardReporting:
    def test_report_counts_by_reason(self):
        g = CounterfactualGate(GateConfig(noise_floor=0.5))
        candidates = [
            cand(list(range(100, 112))),          # kept (delta below set later)
            cand([1]),                             # task drift
            cand(list(range(200, 211))),           # alignment (11 tokens)
        ]
        deltas = {0: 1.0, 1: 1.0, 2: 1.0}
        calls = {"i": -1}

        def measure(_c):
            calls["i"] += 1
            return deltas[calls["i"]]

        res = g.evaluate(CLEAN, candidates, measure)
        report = res.discard_report()
        assert report["n_candidates"] == 3
        assert report["n_valid"] == 1
        assert report["discarded_by_reason"]["task_drift"] == 1
        assert report["discarded_by_reason"]["alignment"] == 1

    def test_all_discarded_flags_insufficient(self):
        g = CounterfactualGate(GateConfig(noise_floor=10.0))
        res = g.evaluate(CLEAN, [cand(list(range(100, 112)))],
                         measure_const(0.1))
        assert res.sufficient is False
        assert res.discard_report()["sufficient"] is False

    def test_report_is_json_safe(self):
        import json
        g = CounterfactualGate(GateConfig())
        res = g.evaluate(CLEAN, [cand([1])], measure_const(1.0))
        json.dumps(res.discard_report())

    def test_strategies_preserved_for_the_technical_file(self):
        g = CounterfactualGate(GateConfig())
        res = g.evaluate(
            CLEAN,
            [cand(list(range(100, 112)), strategy="antonym")],
            measure_const(1.0),
        )
        assert res.valid[0].strategy == "antonym"
