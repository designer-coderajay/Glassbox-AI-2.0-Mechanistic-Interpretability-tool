"""
tests/test_evidence_tier.py — V5 degradation ladder (ROADMAP_V5 Part 6).

The contract: every audit yields a tier; downgrades are never silent;
the engine is pure and deterministic.
"""

import pytest

from glassbox.evidence_tier import (
    EvidenceTier,
    TierEngine,
    TierSignals,
)


def signals(**overrides):
    """Tier-A-capable baseline; tests knock out one capability at a time.

    Removing weight access also removes the capabilities that physically
    depend on it (exact patching, causal abstraction) unless a test sets
    them explicitly — mirroring what any real caller would do.
    """
    base = dict(
        has_weights=True,
        counterfactual_valid=True,
        hessian_reliable=True,
        exact_patch_verified=True,
        causal_abstraction_tested=True,
        behavioral_possible=True,
        sample_n=100,
        min_sample_n=20,
    )
    base.update(overrides)
    if not base["has_weights"]:
        if "exact_patch_verified" not in overrides:
            base["exact_patch_verified"] = False
        if "causal_abstraction_tested" not in overrides:
            base["causal_abstraction_tested"] = False
    return TierSignals(**base)


class TestLadderRules:
    def test_full_evidence_reaches_tier_a(self):
        a = TierEngine().assess(signals())
        assert a.tier is EvidenceTier.A_CAUSAL_CERTIFIED
        assert a.downgrades == []

    def test_missing_causal_abstraction_caps_at_b(self):
        a = TierEngine().assess(signals(causal_abstraction_tested=False))
        assert a.tier is EvidenceTier.B_CAUSAL_SCREENED

    def test_missing_exact_patching_caps_at_b(self):
        a = TierEngine().assess(signals(exact_patch_verified=False))
        assert a.tier is EvidenceTier.B_CAUSAL_SCREENED

    def test_unreliable_hessian_caps_at_c(self):
        a = TierEngine().assess(signals(exact_patch_verified=False,
                                        hessian_reliable=False))
        assert a.tier is EvidenceTier.C_BEHAVIORAL

    def test_invalid_counterfactuals_cap_at_c(self):
        a = TierEngine().assess(signals(counterfactual_valid=False))
        assert a.tier is EvidenceTier.C_BEHAVIORAL

    def test_unverified_counterfactuals_treated_as_not_valid(self):
        a = TierEngine().assess(signals(counterfactual_valid=None))
        assert a.tier is EvidenceTier.C_BEHAVIORAL

    def test_no_weights_caps_at_c(self):
        a = TierEngine().assess(signals(has_weights=False))
        assert a.tier is EvidenceTier.C_BEHAVIORAL

    def test_nothing_available_is_descriptive_not_an_error(self):
        a = TierEngine().assess(signals(has_weights=False,
                                        behavioral_possible=False))
        assert a.tier is EvidenceTier.D_DESCRIPTIVE

    def test_underpowered_sample_downgrades_one_tier_with_reason(self):
        a = TierEngine().assess(signals(sample_n=5))
        assert a.tier is EvidenceTier.B_CAUSAL_SCREENED
        assert any("sample" in r.lower() for r in a.reasons)

    def test_underpowered_sample_from_b_lands_at_c(self):
        a = TierEngine().assess(signals(exact_patch_verified=False, sample_n=5))
        assert a.tier is EvidenceTier.C_BEHAVIORAL


class TestNeverSilent:
    def test_every_downgrade_carries_a_reason(self):
        a = TierEngine().assess(
            signals(causal_abstraction_tested=False, sample_n=3)
        )
        assert len(a.downgrades) >= 1
        for step in a.downgrades:
            assert step.reason and isinstance(step.reason, str)

    def test_disclosure_text_names_tier_and_reasons(self):
        a = TierEngine().assess(signals(has_weights=False))
        text = a.disclosure_text()
        assert "C" in text
        assert "weight" in text.lower()

    def test_tier_a_disclosure_still_states_method(self):
        text = TierEngine().assess(signals()).disclosure_text()
        assert "A" in text


class TestDeterminismAndSerialization:
    def test_same_signals_same_assessment(self):
        s = signals(exact_patch_verified=False)
        assert TierEngine().assess(s).to_dict() == TierEngine().assess(s).to_dict()

    def test_to_dict_is_json_safe(self):
        import json
        payload = TierEngine().assess(signals(sample_n=None)).to_dict()
        json.dumps(payload)  # must not raise

    def test_grade_letters_are_stable_api(self):
        assert EvidenceTier.A_CAUSAL_CERTIFIED.grade == "A"
        assert EvidenceTier.B_CAUSAL_SCREENED.grade == "B"
        assert EvidenceTier.C_BEHAVIORAL.grade == "C"
        assert EvidenceTier.D_DESCRIPTIVE.grade == "D"

    def test_invalid_signal_combo_rejected(self):
        # exact patch verification claimed without weights is contradictory
        with pytest.raises(ValueError):
            TierEngine().assess(signals(has_weights=False,
                                        exact_patch_verified=True))
