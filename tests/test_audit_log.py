"""
tests/test_audit_log.py — Unit tests for glassbox.audit_log (AuditRecord, AuditLog).

Tests run WITHOUT a live model — we use synthetic result dicts that match
the schema produced by GlassboxV2.analyze() and REST API responses.
"""

import csv
import json
import os
import tempfile
import time
import uuid
import pytest

from glassbox.audit_log import AuditRecord, AuditLog


# ---------------------------------------------------------------------------
# Fixtures — synthetic result dicts and test data
# ---------------------------------------------------------------------------


def _make_glassbox_result(
    model_name="test-model",
    analysis_mode="white_box",
    suff=0.85,
    comp=0.62,
    f1=None,
    grade="A",
    compliance_status="conditionally_compliant",
    n_circuit=4,
):
    """Build a realistic GlassboxV2.analyze() result dict."""
    if f1 is None:
        f1 = 2 * suff * comp / (suff + comp) if (suff + comp) > 0 else 0.0

    circuit = [f"L{i}H{j}" for i in range(2) for j in range(n_circuit // 2)]
    return {
        "model_name": model_name,
        "analysis_mode": analysis_mode,
        "prompt": "When Mary and John went to the store, Mary gave a drink to",
        "correct_token": " Mary",
        "incorrect_token": " John",
        "explainability_grade": f"{grade} — Explainability verified",
        "compliance_status": compliance_status,
        "report_id": "GB-12345678",
        "faithfulness": {
            "sufficiency": suff,
            "comprehensiveness": comp,
            "f1": f1,
            "category": "faithful" if f1 > 0.7 else "moderate",
        },
        "full_report": {
            "provider_name": "Test Provider",
            "deployment_context": "financial_services",
            "compliance_status": compliance_status,
            "sections": {
                "1_general_description": {
                    "provider_name": "Test Provider",
                    "deployment_context": "financial_services",
                },
                "2_development_design": {
                    "circuit_heads": circuit,
                    "attribution_scores": {c: 0.25 for c in circuit},
                    "n_layers": 12,
                    "n_heads": 12,
                },
                "3_monitoring_control": {
                    "f1_score": f1,
                    "sufficiency": suff,
                    "comprehensiveness": comp,
                    "explainability_grade": f"{grade} — Explainability verified",
                },
            },
        },
    }


def _make_rest_api_result(
    model_name="api-model",
    grade="B",
    f1=0.65,
    sufficiency=0.75,
    comprehensiveness=0.60,
):
    """Build a REST API response dict (different schema).

    The append_from_result method expects a "faithfulness" dict with f1, sufficiency, comprehensiveness.
    """
    return {
        "model_name": model_name,
        "target_model": model_name,
        "analysis_mode": "black_box",
        "decision_prompt": "Test prompt for API",
        "explainability_grade": f"{grade} — Grade {grade}",
        "compliance_status": "incomplete",
        "circuit": ["L0H0", "L1H1"],
        "faithfulness": {
            "f1": f1,
            "sufficiency": sufficiency,
            "comprehensiveness": comprehensiveness,
        },
    }


@pytest.fixture
def tmp_log_path(tmp_path):
    """Provide a temporary path for a log file."""
    return tmp_path / "audit.jsonl"


@pytest.fixture
def good_result():
    """High faithfulness result — Grade A."""
    return _make_glassbox_result(suff=0.92, comp=0.68, grade="A")


@pytest.fixture
def medium_result():
    """Medium faithfulness — Grade B."""
    return _make_glassbox_result(suff=0.71, comp=0.44, grade="B")


@pytest.fixture
def bad_result():
    """Low faithfulness — Grade D."""
    return _make_glassbox_result(suff=0.35, comp=0.18, grade="D", f1=0.42)


@pytest.fixture
def api_result():
    """REST API response dict."""
    return _make_rest_api_result()


# ---------------------------------------------------------------------------
# Test: AuditRecord construction and hashing
# ---------------------------------------------------------------------------


def test_audit_record_construction():
    """Test basic AuditRecord instantiation."""
    rec = AuditRecord(
        record_id="ABCD1234",
        timestamp_utc=1234567890.0,
        model_name="test-model",
        analysis_mode="white_box",
        prompt="test prompt",
        correct_token=" correct",
        incorrect_token=" wrong",
        provider_name="Test Inc",
        deployment_context="financial_services",
        explainability_grade="A",
        compliance_status="conditionally_compliant",
        faithfulness_f1=0.85,
        faithfulness_sufficiency=0.90,
        faithfulness_comprehensiveness=0.80,
        n_circuit_heads=8,
        report_id="GB-ABC123",
    )
    assert rec.record_id == "ABCD1234"
    assert rec.model_name == "test-model"
    assert rec.explainability_grade == "A"
    assert rec.record_hash != ""  # hash computed in __post_init__
    assert len(rec.record_hash) == 64  # SHA-256 hex is 64 chars


def test_audit_record_hash_computed():
    """Test that hash is automatically computed in __post_init__."""
    rec = AuditRecord(
        record_id="REC1",
        timestamp_utc=time.time(),
        model_name="model",
        analysis_mode="white_box",
        prompt="p",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="A",
        compliance_status="compliant",
        faithfulness_f1=0.8,
        faithfulness_sufficiency=0.8,
        faithfulness_comprehensiveness=0.8,
        n_circuit_heads=1,
        report_id="R1",
        prev_hash="",
    )
    assert rec.record_hash != ""
    assert rec.verify() is True


def test_audit_record_verify_valid():
    """Test that verify() returns True for a valid record."""
    rec = AuditRecord(
        record_id="VALID",
        timestamp_utc=1000000.0,
        model_name="m",
        analysis_mode="white_box",
        prompt="p",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="B",
        compliance_status="non_compliant",
        faithfulness_f1=0.5,
        faithfulness_sufficiency=0.5,
        faithfulness_comprehensiveness=0.5,
        n_circuit_heads=2,
        report_id="R",
        prev_hash="",
    )
    assert rec.verify() is True


def test_audit_record_verify_tampered():
    """Test that verify() returns False when a field is tampered with."""
    rec = AuditRecord(
        record_id="TAMPER",
        timestamp_utc=1000000.0,
        model_name="m",
        analysis_mode="white_box",
        prompt="p",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="A",
        compliance_status="compliant",
        faithfulness_f1=0.8,
        faithfulness_sufficiency=0.8,
        faithfulness_comprehensiveness=0.8,
        n_circuit_heads=4,
        report_id="R",
        prev_hash="",
    )
    original_hash = rec.record_hash
    assert rec.verify() is True

    # Tamper with a canonical field
    rec.model_name = "different_model"
    assert rec.verify() is False

    # Tamper with non-canonical field (should not affect verify)
    rec.model_name = "m"  # restore
    rec.notes = "tampered notes"
    assert rec.verify() is True  # notes are not in canonical hash


def test_audit_record_to_dict():
    """Test that to_dict() returns a complete dict representation."""
    rec = AuditRecord(
        record_id="DICT1",
        timestamp_utc=1234567890.0,
        model_name="test",
        analysis_mode="black_box",
        prompt="prompt",
        correct_token="c",
        incorrect_token="i",
        provider_name="p",
        deployment_context="financial_services",
        explainability_grade="C",
        compliance_status="incomplete",
        faithfulness_f1=0.6,
        faithfulness_sufficiency=0.65,
        faithfulness_comprehensiveness=0.55,
        n_circuit_heads=5,
        report_id="GB-123",
        auditor="test@example.com",
        notes="test notes",
        prev_hash="abc123",
    )
    d = rec.to_dict()
    assert isinstance(d, dict)
    assert d["record_id"] == "DICT1"
    assert d["model_name"] == "test"
    assert d["record_hash"] == rec.record_hash
    assert d["prev_hash"] == "abc123"
    assert d["auditor"] == "test@example.com"


# ---------------------------------------------------------------------------
# Test: Hash chain integrity
# ---------------------------------------------------------------------------


def test_hash_chain_first_record_empty_prev():
    """Test that the first record in a chain has prev_hash=''."""
    rec = AuditRecord(
        record_id="FIRST",
        timestamp_utc=time.time(),
        model_name="m",
        analysis_mode="white_box",
        prompt="p",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="A",
        compliance_status="compliant",
        faithfulness_f1=0.9,
        faithfulness_sufficiency=0.9,
        faithfulness_comprehensiveness=0.9,
        n_circuit_heads=1,
        report_id="R",
        prev_hash="",
    )
    assert rec.prev_hash == ""


def test_hash_chain_sequential():
    """Test that each record's prev_hash links to the previous record."""
    rec1 = AuditRecord(
        record_id="REC1",
        timestamp_utc=1000000.0,
        model_name="m1",
        analysis_mode="white_box",
        prompt="p1",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="A",
        compliance_status="compliant",
        faithfulness_f1=0.8,
        faithfulness_sufficiency=0.8,
        faithfulness_comprehensiveness=0.8,
        n_circuit_heads=1,
        report_id="R1",
        prev_hash="",
    )

    rec2 = AuditRecord(
        record_id="REC2",
        timestamp_utc=1000001.0,
        model_name="m2",
        analysis_mode="white_box",
        prompt="p2",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="B",
        compliance_status="non_compliant",
        faithfulness_f1=0.6,
        faithfulness_sufficiency=0.6,
        faithfulness_comprehensiveness=0.6,
        n_circuit_heads=2,
        report_id="R2",
        prev_hash=rec1.record_hash,  # Links to rec1
    )

    assert rec2.prev_hash == rec1.record_hash
    assert rec2.verify() is True


# ---------------------------------------------------------------------------
# Test: AuditLog append and persistence
# ---------------------------------------------------------------------------


def test_audit_log_append(tmp_log_path):
    """Test that append() creates a record and writes to disk."""
    log = AuditLog(str(tmp_log_path))
    assert len(log) == 0

    rec = log.append(
        model_name="test-model",
        analysis_mode="white_box",
        prompt="test prompt",
        correct_token=" correct",
        incorrect_token=" wrong",
        provider_name="Test Inc",
        deployment_context="financial_services",
        explainability_grade="A",
        compliance_status="conditionally_compliant",
        faithfulness_f1=0.85,
        faithfulness_sufficiency=0.90,
        faithfulness_comprehensiveness=0.80,
        n_circuit_heads=8,
        report_id="GB-ABC123",
        auditor="test@example.com",
    )

    assert rec.record_id is not None
    assert rec.model_name == "test-model"
    assert len(log) == 1
    assert tmp_log_path.exists()


def test_audit_log_append_chain(tmp_log_path):
    """Test that consecutive appends form a hash chain."""
    log = AuditLog(str(tmp_log_path))

    rec1 = log.append(
        model_name="model1",
        analysis_mode="white_box",
        prompt="p1",
        explainability_grade="A",
        compliance_status="conditionally_compliant",
        faithfulness_f1=0.9,
    )

    rec2 = log.append(
        model_name="model2",
        analysis_mode="white_box",
        prompt="p2",
        explainability_grade="B",
        compliance_status="non_compliant",
        faithfulness_f1=0.6,
    )

    assert rec1.prev_hash == ""
    assert rec2.prev_hash == rec1.record_hash
    assert len(log) == 2


def test_audit_log_append_from_result_glassbox(tmp_log_path, good_result):
    """Test append_from_result() with GlassboxV2-style result dict."""
    log = AuditLog(str(tmp_log_path))
    rec = log.append_from_result(good_result, auditor="auditor@test.com", notes="Test audit")

    assert rec.model_name == "test-model"
    assert rec.analysis_mode == "white_box"
    assert rec.explainability_grade == "A"
    assert rec.faithfulness_f1 == 0.92 * 0.68 * 2 / (0.92 + 0.68)  # computed f1
    assert rec.faithfulness_sufficiency == 0.92
    assert rec.faithfulness_comprehensiveness == 0.68
    assert rec.auditor == "auditor@test.com"
    assert rec.notes == "Test audit"


def test_audit_log_append_from_result_rest_api(tmp_log_path, api_result):
    """Test append_from_result() with REST API response dict."""
    log = AuditLog(str(tmp_log_path))
    rec = log.append_from_result(api_result, auditor="api_user@test.com")

    assert rec.model_name == "api-model"
    assert rec.analysis_mode == "black_box"
    assert rec.explainability_grade == "B"
    assert rec.faithfulness_f1 == 0.65
    assert rec.faithfulness_sufficiency == 0.75
    assert rec.faithfulness_comprehensiveness == 0.60


def test_audit_log_persistence(tmp_log_path, good_result, medium_result):
    """Test that records are persisted and reloaded correctly."""
    # Create and populate log
    log1 = AuditLog(str(tmp_log_path))
    log1.append_from_result(good_result, auditor="user1")
    log1.append_from_result(medium_result, auditor="user2")
    assert len(log1) == 2

    # Load log from same path
    log2 = AuditLog(str(tmp_log_path))
    assert len(log2) == 2
    assert log2.records()[0].model_name == "test-model"
    assert log2.records()[1].model_name == "test-model"


# ---------------------------------------------------------------------------
# Test: AuditLog verification
# ---------------------------------------------------------------------------


def test_verify_chain_valid(tmp_log_path, good_result, medium_result):
    """Test that verify_chain() returns True for an untampered log."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)
    log.append_from_result(medium_result)

    assert log.verify_chain() is True


def test_verify_chain_invalid_after_corruption(tmp_log_path, good_result, medium_result):
    """Test that verify_chain() returns False when a line is corrupted."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)
    log.append_from_result(medium_result)

    # Corrupt the first record by modifying its hash in the file
    with open(tmp_log_path, "r") as f:
        lines = f.readlines()
    d = json.loads(lines[0])
    d["record_hash"] = "corrupted_hash"
    lines[0] = json.dumps(d) + "\n"

    with open(tmp_log_path, "w") as f:
        f.writelines(lines)

    # Reload and verify
    log2 = AuditLog(str(tmp_log_path))
    assert log2.verify_chain() is False


# ---------------------------------------------------------------------------
# Test: AuditLog query methods
# ---------------------------------------------------------------------------


def test_audit_log_records(tmp_log_path, good_result, medium_result, bad_result):
    """Test records() returns all records in order."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)
    log.append_from_result(medium_result)
    log.append_from_result(bad_result)

    recs = log.records()
    assert len(recs) == 3
    assert recs[0].explainability_grade == "A"
    assert recs[1].explainability_grade == "B"
    assert recs[2].explainability_grade == "D"


def test_audit_log_latest(tmp_log_path, good_result):
    """Test latest(n) returns the N most recent records."""
    log = AuditLog(str(tmp_log_path))
    for i in range(5):
        result = _make_glassbox_result(model_name=f"model{i}")
        log.append_from_result(result)

    latest = log.latest(3)
    assert len(latest) == 3
    assert latest[0].model_name == "model2"
    assert latest[2].model_name == "model4"


def test_audit_log_by_model(tmp_log_path):
    """Test by_model() filters records by model name."""
    log = AuditLog(str(tmp_log_path))
    result_a = _make_glassbox_result(model_name="modelA")
    result_b = _make_glassbox_result(model_name="modelB")
    log.append_from_result(result_a)
    log.append_from_result(result_b)
    log.append_from_result(result_a)

    by_a = log.by_model("modelA")
    assert len(by_a) == 2
    for rec in by_a:
        assert rec.model_name == "modelA"


def test_audit_log_by_grade(tmp_log_path, good_result, medium_result, bad_result):
    """Test by_grade() filters records by explainability grade."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)
    log.append_from_result(medium_result)
    log.append_from_result(bad_result)
    log.append_from_result(good_result)

    by_a = log.by_grade("A")
    assert len(by_a) == 2
    for rec in by_a:
        assert rec.explainability_grade == "A"


def test_audit_log_non_compliant(tmp_log_path):
    """Test non_compliant() filters by compliance status."""
    log = AuditLog(str(tmp_log_path))
    result_compliant = _make_glassbox_result(compliance_status="conditionally_compliant")
    result_non_compliant = _make_glassbox_result(compliance_status="non_compliant")
    log.append_from_result(result_compliant)
    log.append_from_result(result_non_compliant)
    log.append_from_result(result_non_compliant)

    nc = log.non_compliant()
    assert len(nc) == 2
    for rec in nc:
        assert rec.compliance_status == "non_compliant"


# ---------------------------------------------------------------------------
# Test: AuditLog summary
# ---------------------------------------------------------------------------


def test_summary_empty_log(tmp_log_path):
    """Test summary() on an empty log."""
    log = AuditLog(str(tmp_log_path))
    summary = log.summary()

    assert summary["total_audits"] == 0
    assert summary["grade_distribution"] == {"A": 0, "B": 0, "C": 0, "D": 0}
    assert summary["compliance_rate"] == 0.0
    assert summary["avg_f1"] == 0.0
    assert summary["models_audited"] == []
    assert summary["non_compliant_count"] == 0
    assert summary["latest_audit_utc"] is None
    assert summary["chain_valid"] is True


def test_summary_populated_log(tmp_log_path):
    """Test summary() returns correct aggregates."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(_make_glassbox_result(grade="A", suff=0.9, comp=0.85))
    log.append_from_result(_make_glassbox_result(grade="B", suff=0.7, comp=0.6))
    log.append_from_result(_make_glassbox_result(grade="D", suff=0.3, comp=0.2, compliance_status="non_compliant"))

    summary = log.summary()
    assert summary["total_audits"] == 3
    assert summary["grade_distribution"]["A"] == 1
    assert summary["grade_distribution"]["B"] == 1
    assert summary["grade_distribution"]["D"] == 1
    assert summary["non_compliant_count"] == 1
    assert summary["compliance_rate"] == pytest.approx(2.0 / 3, abs=0.01)
    assert summary["avg_f1"] > 0
    assert summary["models_audited"] == ["test-model"]
    assert summary["latest_audit_utc"] is not None
    assert summary["chain_valid"] is True


def test_summary_multiple_models(tmp_log_path):
    """Test summary() with multiple distinct models."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(_make_glassbox_result(model_name="modelX", grade="A"))
    log.append_from_result(_make_glassbox_result(model_name="modelY", grade="B"))
    log.append_from_result(_make_glassbox_result(model_name="modelZ", grade="C"))

    summary = log.summary()
    assert summary["total_audits"] == 3
    assert set(summary["models_audited"]) == {"modelX", "modelY", "modelZ"}


# ---------------------------------------------------------------------------
# Test: Export to CSV and JSON
# ---------------------------------------------------------------------------


def test_export_csv(tmp_log_path, good_result, medium_result):
    """Test export_csv() writes correct CSV file."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result, auditor="user@test.com")
    log.append_from_result(medium_result, auditor="user2@test.com")

    csv_path = tmp_log_path.parent / "export.csv"
    log.export_csv(str(csv_path))

    assert csv_path.exists()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["model_name"] == "test-model"
    assert rows[0]["explainability_grade"] == "A"
    assert rows[0]["auditor"] == "user@test.com"


def test_export_json(tmp_log_path, good_result):
    """Test export_json() writes correct JSON structure."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result, auditor="test@example.com")

    json_path = tmp_log_path.parent / "export.json"
    result_json = log.export_json(str(json_path))

    # Test return value
    data = json.loads(result_json)
    assert data["glassbox_audit_log"]["version"] == "1.0"
    assert data["glassbox_audit_log"]["total_records"] == 1
    assert data["glassbox_audit_log"]["chain_valid"] is True
    assert len(data["glassbox_audit_log"]["records"]) == 1

    # Test file written
    assert json_path.exists()
    with open(json_path) as f:
        file_data = json.load(f)
    assert file_data == data


def test_export_json_returns_string(tmp_log_path, good_result):
    """Test that export_json() returns JSON string when no path given."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)

    result_str = log.export_json()
    assert isinstance(result_str, str)
    data = json.loads(result_str)
    assert "glassbox_audit_log" in data


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


def test_audit_log_empty_log_repr(tmp_log_path):
    """Test __repr__ on empty log."""
    log = AuditLog(str(tmp_log_path))
    r = repr(log)
    assert "AuditLog" in r
    assert "records=0" in r


def test_audit_log_non_empty_log_repr(tmp_log_path, good_result):
    """Test __repr__ on non-empty log."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)
    r = repr(log)
    assert "AuditLog" in r
    assert "records=1" in r


def test_audit_log_prompt_truncation(tmp_log_path):
    """Test that append() truncates long prompts to 500 chars."""
    log = AuditLog(str(tmp_log_path))
    long_prompt = "x" * 1000
    rec = log.append(
        model_name="m",
        analysis_mode="white_box",
        prompt=long_prompt,
        explainability_grade="A",
        compliance_status="compliant",
    )
    assert len(rec.prompt) == 500


def test_audit_log_float_rounding(tmp_log_path):
    """Test that faithfulness scores are rounded to 6 decimals."""
    log = AuditLog(str(tmp_log_path))
    rec = log.append(
        model_name="m",
        analysis_mode="white_box",
        prompt="p",
        faithfulness_f1=0.123456789,
        faithfulness_sufficiency=0.987654321,
        faithfulness_comprehensiveness=0.555555555,
        explainability_grade="A",
        compliance_status="compliant",
    )
    assert rec.faithfulness_f1 == round(0.123456789, 6)
    assert rec.faithfulness_sufficiency == round(0.987654321, 6)
    assert rec.faithfulness_comprehensiveness == round(0.555555555, 6)


def test_audit_record_default_values():
    """Test that optional fields have sensible defaults."""
    rec = AuditRecord(
        record_id="R",
        timestamp_utc=0.0,
        model_name="m",
        analysis_mode="white_box",
        prompt="p",
        correct_token="c",
        incorrect_token="i",
        provider_name="prov",
        deployment_context="ctx",
        explainability_grade="D",
        compliance_status="non_compliant",
        faithfulness_f1=0.0,
        faithfulness_sufficiency=0.0,
        faithfulness_comprehensiveness=0.0,
        n_circuit_heads=0,
        report_id="",
    )
    assert rec.auditor == ""
    assert rec.notes == ""
    assert rec.prev_hash == ""


def test_audit_log_reload_with_blank_lines(tmp_log_path, good_result):
    """Test that AuditLog._load() skips blank lines gracefully."""
    log = AuditLog(str(tmp_log_path))
    log.append_from_result(good_result)

    # Add some blank lines to the JSONL
    with open(tmp_log_path, "a") as f:
        f.write("\n\n")

    # Reload
    log2 = AuditLog(str(tmp_log_path))
    assert len(log2) == 1  # Should skip blank lines


def test_audit_log_length_operator(tmp_log_path, good_result):
    """Test that len() operator works correctly."""
    log = AuditLog(str(tmp_log_path))
    assert len(log) == 0
    log.append_from_result(good_result)
    assert len(log) == 1
    log.append_from_result(good_result)
    assert len(log) == 2
