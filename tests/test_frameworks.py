"""Tests for glassbox/frameworks.py — NIST AI RMF + ISO 42001 cross-walk."""
from glassbox.frameworks import ISO_42001_OBJECTIVES, NIST_AI_RMF, framework_pack


def test_framework_constants():
    assert NIST_AI_RMF == ["GOVERN", "MAP", "MEASURE", "MANAGE"]
    assert len(ISO_42001_OBJECTIVES) == 9  # A.2 .. A.10
    assert ISO_42001_OBJECTIVES["A.6"].lower().startswith("ai system life")


def test_pack_maps_evidence_to_both_frameworks():
    pack = framework_pack(["faithfulness", "annex_iv", "drift_monitoring"])
    # faithfulness -> MEASURE; annex_iv -> MAP; drift -> MANAGE
    assert "MEASURE" in pack["nist_ai_rmf"]
    assert "MAP" in pack["nist_ai_rmf"]
    assert "MANAGE" in pack["nist_ai_rmf"]
    measure_ev = {e["evidence"] for e in pack["nist_ai_rmf"]["MEASURE"]}
    assert "faithfulness" in measure_ev
    # iso objectives populated
    assert "A.6" in pack["iso_42001"]
    assert set(pack["evidence_used"]) == {"faithfulness", "annex_iv", "drift_monitoring"}
    assert "not certification" in pack["disclaimer"]


def test_pack_ignores_unknown_evidence():
    pack = framework_pack(["faithfulness", "totally_made_up"])
    assert pack["evidence_used"] == ["faithfulness"]


def test_empty_pack():
    pack = framework_pack([])
    assert pack["nist_ai_rmf"] == {}
    assert pack["iso_42001"] == {}
    assert pack["evidence_used"] == []
    assert pack["disclaimer"]  # disclaimer always present


def test_full_evidence_covers_all_nist_functions():
    pack = framework_pack([
        "faithfulness", "evidence_tier", "annex_iv",
        "risk_management", "drift_monitoring", "confidence_gap", "data_governance",
    ])
    # GOVERN (tier), MAP (annex_iv/risk/data), MEASURE (faithfulness/tier/conf), MANAGE (risk/drift)
    assert set(pack["nist_ai_rmf"].keys()) == set(NIST_AI_RMF)
