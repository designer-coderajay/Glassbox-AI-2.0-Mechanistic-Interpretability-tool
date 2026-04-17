"""
tests/test_compliance.py — Unit tests for glassbox.compliance (AnnexIVReport).

Tests run WITHOUT a live model — we use a realistic synthetic analyze() result
that matches the exact structure produced by GlassboxV2.analyze().
"""

import json
import os
import tempfile
import pytest

from glassbox.compliance import (
    AnnexIVReport,
    DeploymentContext,
    RiskClassification,
    ExplainabilityGrade,
    ComplianceStatus,
    Section1_GeneralDescription,
    Section3_MonitoringControl,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic analyze() results (match GlassboxV2.analyze() schema)
# ---------------------------------------------------------------------------

def _make_result(suff=0.85, comp=0.62, f1=None, method="taylor", n_circuit=4):
    """Build a realistic GlassboxV2.analyze() result dict."""
    if f1 is None:
        f1 = 2 * suff * comp / (suff + comp) if (suff + comp) > 0 else 0.0

    if suff > 0.9 and comp < 0.4:
        category = "backup_mechanisms"
    elif suff > 0.7 and comp > 0.5:
        category = "faithful"
    elif suff < 0.5:
        category = "incomplete"
    elif suff < 0.6 and comp < 0.5:
        category = "weak"
    else:
        category = "moderate"

    circuit = [(i, j) for i in range(2) for j in range(n_circuit // 2)]
    return {
        "circuit":    circuit,
        "n_heads":    len(circuit),
        "clean_ld":   3.14,
        "corr_prompt": "When John and Mary went to the store, Mary gave a drink to",
        "attributions": {str(h): 0.25 for h in circuit},
        "mlp_attributions": {"0": 0.1, "1": 0.05},
        "top_heads": [
            {"layer": h[0], "head": h[1], "attr": 0.25, "rel_depth": h[0] / 11}
            for h in circuit
        ],
        "method": method,
        "faithfulness": {
            "sufficiency":       suff,
            "comprehensiveness": comp,
            "f1":                f1,
            "category":          category,
            "suff_is_approx":    method == "taylor",
        },
        "model_metadata": {
            "model_name":       "gpt2",
            "n_layers":         12,
            "n_heads":          12,
            "d_model":          768,
            "d_head":           64,
            "glassbox_version": "2.6.0",
        },
    }


@pytest.fixture
def good_result():
    """High faithfulness result — Grade A.
    suff=0.92, comp=0.72 → F1=0.826 ≥ 0.80 threshold for Grade A.
    (comp=0.68 only gives F1=0.782 which lands in Grade B.)
    """
    return _make_result(suff=0.92, comp=0.72)


@pytest.fixture
def medium_result():
    """Medium faithfulness — Grade B/C."""
    return _make_result(suff=0.71, comp=0.44)


@pytest.fixture
def bad_result():
    """Low faithfulness — Grade D, risk flag."""
    return _make_result(suff=0.35, comp=0.18)


@pytest.fixture
def report_fintech(good_result):
    """Standard fintech report with one good analysis."""
    r = AnnexIVReport(
        model_name         = "CreditScorer v3.2",
        system_purpose     = "Credit risk assessment for loan applications",
        provider_name      = "Acme Bank NV",
        provider_address   = "1 Fintech Street, Amsterdam 1011AB, Netherlands",
        deployment_context = DeploymentContext.FINANCIAL_SERVICES,
    )
    r.add_analysis(good_result, use_case="Loan denial — insufficient credit history")
    return r


# ---------------------------------------------------------------------------
# Test: instantiation
# ---------------------------------------------------------------------------

def test_report_instantiation():
    r = AnnexIVReport(
        model_name         = "TestModel",
        system_purpose     = "Test purpose",
        provider_name      = "Test Corp",
        provider_address   = "1 Test St",
        deployment_context = DeploymentContext.HEALTHCARE,
    )
    assert r.model_name == "TestModel"
    assert r.deployment_context == DeploymentContext.HEALTHCARE
    assert r.risk_classification == RiskClassification.HIGH_RISK  # default
    assert len(r._analyses) == 0


def test_method_chaining(good_result, medium_result):
    r = AnnexIVReport(
        model_name="M", system_purpose="P",
        provider_name="N", provider_address="A",
    )
    result = r.add_analysis(good_result).add_analysis(medium_result)
    assert result is r  # returns self
    assert len(r._analyses) == 2


# ---------------------------------------------------------------------------
# Test: section population
# ---------------------------------------------------------------------------

def test_section1_populated(report_fintech):
    s1 = report_fintech._s1
    assert s1 is not None
    assert s1.system_name    == "CreditScorer v3.2"
    assert s1.provider_name  == "Acme Bank NV"
    assert s1.model_n_layers == 12
    assert s1.model_n_heads  == 12
    assert s1.model_d_model  == 768
    assert s1.deployment_context == "financial_services"
    assert s1.geographic_scope   == "European Union"
    assert len(s1.report_id) == 8


def test_section2_populated(report_fintech):
    s2 = report_fintech._s2
    assert s2 is not None
    assert s2.n_analysis_heads_total == 144  # 12 layers * 12 heads
    assert s2.n_circuit_heads_found  >= 1
    assert s2.attribution_method     == "taylor"
    assert len(s2.reference_papers)  >= 4
    assert "Nanda" in s2.reference_papers[0]


def test_section3_faithfulness_metrics(report_fintech):
    s3 = report_fintech._s3
    assert s3 is not None
    assert 0.0 <= s3.sufficiency <= 1.0
    assert 0.0 <= s3.comprehensiveness <= 1.0
    assert 0.0 <= s3.f1_score <= 1.0
    assert s3.faithfulness_category in {"faithful", "backup_mechanisms", "incomplete", "weak", "moderate"}
    assert s3.explainability_grade != ""
    assert len(s3.explainability_rationale) > 50


def test_all_nine_sections_populated(report_fintech):
    for attr in ["_s1","_s2","_s3","_s4","_s5","_s6","_s7","_s8","_s9"]:
        assert getattr(report_fintech, attr) is not None, f"Section {attr} not populated"


# ---------------------------------------------------------------------------
# Test: explainability grading
# ---------------------------------------------------------------------------

def test_grade_a(good_result):
    r = AnnexIVReport("M","P","N","A")
    r.add_analysis(good_result)
    assert r._s3.explainability_grade.startswith("A")


def test_grade_d_risk_flag(bad_result):
    r = AnnexIVReport("M","P","N","A")
    r.add_analysis(bad_result)
    assert r._s5.faithfulness_risk_flag is True
    assert r._s3.explainability_grade.startswith("D")


def test_grade_b(medium_result):
    r = AnnexIVReport("M","P","N","A")
    r.add_analysis(medium_result)
    grade = r._s3.explainability_grade
    assert grade.startswith("B") or grade.startswith("C")


# ---------------------------------------------------------------------------
# Test: JSON output
# ---------------------------------------------------------------------------

def test_json_output_structure(report_fintech):
    jstr = report_fintech.to_json()
    data = json.loads(jstr)

    assert data["document_type"]     == "EU AI Act Annex IV Technical Documentation"
    assert data["regulation"]        == "Regulation (EU) 2024/1689 of the European Parliament and of the Council"
    assert data["regulation_article"] == "Article 11 — Technical Documentation"
    assert len(data["report_id"]) == 8

    sections = data["sections"]
    assert len(sections) == 9
    expected = [
        "1_general_description", "2_development_design", "3_monitoring_control",
        "4_data_governance", "5_risk_management", "6_lifecycle_changes",
        "7_harmonised_standards", "8_declaration", "9_post_market_monitoring",
    ]
    for key in expected:
        assert key in sections, f"Missing section: {key}"


def test_json_regulation_refs(report_fintech):
    data = json.loads(report_fintech.to_json())
    for key, section in data["sections"].items():
        assert "_regulation_ref" in section, f"Missing regulation_ref in {key}"
        assert "Annex IV" in section["_regulation_ref"]


def test_json_faithfulness_accuracy(report_fintech, good_result):
    data  = json.loads(report_fintech.to_json())
    s3    = data["sections"]["3_monitoring_control"]
    # Must match source data to 4 decimal places
    orig = good_result["faithfulness"]
    assert abs(s3["sufficiency"]       - orig["sufficiency"])       < 1e-4
    assert abs(s3["comprehensiveness"] - orig["comprehensiveness"]) < 1e-4
    assert abs(s3["f1_score"]          - orig["f1"])                < 1e-4


def test_json_raw_analyses(good_result):
    r = AnnexIVReport("M","P","N","A")
    r.add_analysis(good_result, use_case="Loan denial")
    r.add_analysis(_make_result(suff=0.6, comp=0.4), use_case="Credit check")
    data = json.loads(r.to_json())
    assert data["n_analyses_included"] == 2
    assert len(data["raw_analyses"])   == 2
    assert data["raw_analyses"][0]["use_case"] == "Loan denial"


def test_json_save_to_file(report_fintech, tmp_path):
    path = str(tmp_path / "test_report.json")
    report_fintech.save_json(path)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["document_type"] == "EU AI Act Annex IV Technical Documentation"


# ---------------------------------------------------------------------------
# Test: risk identification
# ---------------------------------------------------------------------------

def test_risks_identified_for_bad_analysis(bad_result):
    r = AnnexIVReport("M","P","N","A")
    r.add_analysis(bad_result)
    risks = r._s5.identified_risks
    risk_names = [ri["risk"] for ri in risks]
    assert any("explainability" in n.lower() for n in risk_names)
    assert any("data governance" in n.lower() for n in risk_names)
    assert any("declaration" in n.lower() for n in risk_names)


def test_recommendations_present(report_fintech):
    recs = report_fintech._s5.recommended_actions
    assert len(recs) >= 3
    assert any("PROVIDER TO COMPLETE" in r for r in recs)
    assert any("Article 47" in r or "conformity" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# Test: batch analysis
# ---------------------------------------------------------------------------

def test_batch_analysis_aggregation():
    results = [
        _make_result(suff=0.80, comp=0.60),
        _make_result(suff=0.90, comp=0.65),
        _make_result(suff=0.85, comp=0.58),
    ]
    r = AnnexIVReport("M","P","N","A")
    r.add_batch_analysis(results, use_cases=["UC1","UC2","UC3"])
    assert len(r._analyses) == 3
    # Averaged sufficiency should be ~0.85
    avg_suff = r._s3.sufficiency
    assert abs(avg_suff - 0.85) < 0.01


# ---------------------------------------------------------------------------
# Test: deployment context mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctx,expected_substring", [
    (DeploymentContext.FINANCIAL_SERVICES, "credit"),
    (DeploymentContext.HEALTHCARE,         "patient"),
    (DeploymentContext.HR_EMPLOYMENT,      "applicant"),
    (DeploymentContext.LEGAL,              "legal"),
])
def test_affected_persons_mapping(ctx, expected_substring, good_result):
    r = AnnexIVReport("M","P","N","A", deployment_context=ctx)
    r.add_analysis(good_result)
    persons = r._s1.categories_of_persons.lower()
    assert expected_substring in persons


# ---------------------------------------------------------------------------
# Test: no analysis raises ValueError
# ---------------------------------------------------------------------------

def test_to_json_raises_without_analysis():
    r = AnnexIVReport("M","P","N","A")
    with pytest.raises(ValueError, match="No analysis results"):
        r.to_json()


# ---------------------------------------------------------------------------
# Test: PDF generation (requires reportlab — skip if not installed)
# ---------------------------------------------------------------------------

def test_pdf_generation(report_fintech, tmp_path):
    pytest.importorskip("reportlab")
    path = str(tmp_path / "annex_iv.pdf")
    result_path = report_fintech.to_pdf(path)
    assert result_path.exists()
    assert result_path.stat().st_size > 10_000  # at least 10KB


def test_pdf_without_reportlab_raises(report_fintech, tmp_path, monkeypatch):
    """If reportlab is not installed, to_pdf() raises ImportError."""
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "reportlab":
            raise ImportError("mocked")
        if name.startswith("reportlab"):
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="reportlab"):
        report_fintech.to_pdf(str(tmp_path / "test.pdf"))


# ---------------------------------------------------------------------------
# Test: section 7 standards
# ---------------------------------------------------------------------------

def test_section7_standards_present(report_fintech):
    s7 = report_fintech._s7
    assert len(s7.standards_applied) >= 3
    std_names = [s["standard"] for s in s7.standards_applied]
    assert any("ISO" in s for s in std_names)
    assert s7.standardisation_body_ref != ""


# ---------------------------------------------------------------------------
# Test: section 9 monitoring
# ---------------------------------------------------------------------------

def test_section9_reaudit_triggers(report_fintech):
    s9 = report_fintech._s9
    assert len(s9.reaudit_triggers)      >= 3
    assert len(s9.performance_indicators) >= 4
    assert "Article 73" in s9.incident_reporting_process
