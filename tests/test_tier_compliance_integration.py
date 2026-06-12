"""
tests/test_tier_compliance_integration.py — V5 Sprint 2:
evidence tiers wired into the Annex IV report (ROADMAP_V5 Part 6 → §4).

Contract:
1. add_analysis() optionally accepts an evidence tier (object or dict, or
   embedded in the result under "evidence_tier").
2. The report's JSON exposes "evidence_tier" with grade + disclosure.
3. With multiple analyses, the WEAKEST tier governs (most conservative).
4. Reports without tier information behave byte-for-byte as before
   (backward compatibility).
"""

import json

from glassbox.compliance import AnnexIVReport, DeploymentContext
from glassbox.evidence_tier import TierEngine, TierSignals


def make_result(suff=0.85, comp=0.62):
    f1 = 2 * suff * comp / (suff + comp)
    circuit = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return {
        "circuit": circuit,
        "n_heads": len(circuit),
        "clean_ld": 3.14,
        "corr_prompt": "corrupted",
        "attributions": {str(h): 0.25 for h in circuit},
        "mlp_attributions": {"0": 0.1},
        "top_heads": [
            {"layer": h[0], "head": h[1], "attr": 0.25, "rel_depth": h[0] / 11}
            for h in circuit
        ],
        "method": "taylor",
        "faithfulness": {
            "sufficiency": suff, "comprehensiveness": comp, "f1": f1,
            "category": "moderate", "suff_is_approx": True,
        },
        "model_metadata": {
            "model_name": "gpt2", "n_layers": 12, "n_heads": 12,
            "d_model": 768, "d_head": 64, "glassbox_version": "4.3.1",
        },
    }


def make_report():
    return AnnexIVReport(
        model_name="CreditScorer v3.2",
        system_purpose="Credit risk assessment",
        provider_name="Acme Bank NV",
        provider_address="Amsterdam",
        deployment_context=DeploymentContext.FINANCIAL_SERVICES,
    )


def tier_b():
    return TierEngine().assess(TierSignals(
        has_weights=True, counterfactual_valid=True, hessian_reliable=True,
        exact_patch_verified=False, causal_abstraction_tested=False,
        sample_n=100,
    ))


def tier_c():
    return TierEngine().assess(TierSignals(
        has_weights=False, counterfactual_valid=None, hessian_reliable=None,
    ))


class TestTierInReport:
    def test_tier_object_lands_in_json(self):
        r = make_report().add_analysis(make_result(), evidence_tier=tier_b())
        data = json.loads(r.to_json())
        assert data["evidence_tier"]["tier"] == "B"
        assert "disclosure" in data["evidence_tier"]

    def test_tier_dict_accepted(self):
        r = make_report().add_analysis(
            make_result(), evidence_tier=tier_b().to_dict()
        )
        data = json.loads(r.to_json())
        assert data["evidence_tier"]["tier"] == "B"

    def test_tier_embedded_in_result_is_picked_up(self):
        result = make_result()
        result["evidence_tier"] = tier_c().to_dict()
        r = make_report().add_analysis(result)
        data = json.loads(r.to_json())
        assert data["evidence_tier"]["tier"] == "C"

    def test_weakest_tier_governs_across_analyses(self):
        r = make_report()
        r.add_analysis(make_result(), evidence_tier=tier_b())
        r.add_analysis(make_result(), evidence_tier=tier_c())
        data = json.loads(r.to_json())
        assert data["evidence_tier"]["tier"] == "C"

    def test_disclosure_text_appears_in_explainability_section(self):
        r = make_report().add_analysis(make_result(), evidence_tier=tier_c())
        payload = r.to_json()
        assert "Evidence tier C" in payload


class TestBackwardCompatibility:
    def test_no_tier_no_key(self):
        r = make_report().add_analysis(make_result())
        data = json.loads(r.to_json())
        assert "evidence_tier" not in data

    def test_no_tier_report_unchanged(self):
        """The legacy path must be unaffected by the new parameter."""
        legacy = make_report().add_analysis(make_result())
        data = json.loads(legacy.to_json())
        assert data["sections"]["3_monitoring_control"]
        assert "evidence_tier_disclosure" not in data["sections"]["3_monitoring_control"]

    def test_method_chaining_preserved(self):
        r = make_report()
        assert r.add_analysis(make_result(), evidence_tier=tier_b()) is r
