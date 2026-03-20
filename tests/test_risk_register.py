"""
tests/test_risk_register.py
===========================
Offline tests for glassbox.risk_register.

All tests use tmp_path — no network, no GPU, no model required.
"""

import json
import os
import pytest

from glassbox.risk_register import RiskEntry, RiskRegister


# ─────────────────────────────────────────────────────────────────────────────
# RiskEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskEntry:

    def test_defaults(self):
        e = RiskEntry("some risk")
        assert e.description == "some risk"
        assert e.model_name  == "unknown"
        assert e.severity    == "medium"
        assert e.status      == "open"
        assert e.occurrences == 1
        assert e.notes       == ""
        assert len(e.risk_id) > 0

    def test_custom_fields(self):
        e = RiskEntry(
            "F1 too low", model_name="gpt2",
            article="Article 13", severity="high",
            status="escalated", notes="Flagged by auditor",
        )
        assert e.model_name == "gpt2"
        assert e.article    == "Article 13"
        assert e.severity   == "high"
        assert e.status     == "escalated"
        assert "auditor" in e.notes

    def test_to_dict_roundtrip(self):
        e  = RiskEntry("test", model_name="m1", severity="critical")
        d  = e.to_dict()
        e2 = RiskEntry.from_dict(d)
        assert e2.risk_id     == e.risk_id
        assert e2.description == e.description
        assert e2.model_name  == e.model_name
        assert e2.severity    == e.severity

    def test_repr(self):
        e = RiskEntry("x", severity="low")
        r = repr(e)
        assert "RiskEntry" in r
        assert "low" in r

    def test_risk_id_auto_generated(self):
        a = RiskEntry("a")
        b = RiskEntry("b")
        assert a.risk_id != b.risk_id

    def test_fixed_risk_id(self):
        e = RiskEntry("x", risk_id="fixed-id-123")
        assert e.risk_id == "fixed-id-123"


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — construction & persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskRegisterPersistence:

    def test_empty_register(self, tmp_path):
        rr = RiskRegister(tmp_path / "risks.json")
        assert len(rr) == 0

    def test_creates_file_on_save(self, tmp_path):
        path = tmp_path / "risks.json"
        rr   = RiskRegister(path)
        rr.add("risk 1", model_name="m1")
        assert path.exists()

    def test_reload_persists_entries(self, tmp_path):
        path = tmp_path / "risks.json"
        rr   = RiskRegister(path)
        rr.add("persistent risk", model_name="gpt2", severity="high")
        rr.add("another risk", model_name="gpt2-xl")

        rr2 = RiskRegister(path)
        assert len(rr2) == 2
        descs = {r.description for r in rr2.all_risks()}
        assert "persistent risk" in descs
        assert "another risk"    in descs

    def test_reload_preserves_status(self, tmp_path):
        path = tmp_path / "risks.json"
        rr   = RiskRegister(path)
        e    = rr.add("risk", model_name="m1")
        rr.set_status(e.risk_id, "mitigated", notes="Fixed it")

        rr2   = RiskRegister(path)
        entry = rr2.get(e.risk_id)
        assert entry.status == "mitigated"
        assert "Fixed it" in entry.notes

    def test_reload_preserves_occurrences(self, tmp_path):
        path = tmp_path / "risks.json"
        rr   = RiskRegister(path)
        rr.add("dup", model_name="m")
        rr.add("dup", model_name="m")   # dedup — should bump count
        rr.add("dup", model_name="m")

        rr2  = RiskRegister(path)
        risk = rr2.all_risks()[0]
        assert risk.occurrences == 3

    def test_nonexistent_path_ok(self, tmp_path):
        rr = RiskRegister(tmp_path / "sub" / "deep" / "risks.json")
        rr.add("test")
        assert len(rr) == 1

    def test_corrupted_file_does_not_crash(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        rr   = RiskRegister(path)   # should not raise
        assert len(rr) == 0


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — add & deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskRegisterAdd:

    def test_add_returns_entry(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("test risk", model_name="m1")
        assert isinstance(e, RiskEntry)
        assert e.description == "test risk"

    def test_dedup_same_desc_same_model(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("same risk", model_name="gpt2")
        rr.add("same risk", model_name="gpt2")
        rr.add("same risk", model_name="gpt2")
        assert len(rr) == 1
        assert rr.all_risks()[0].occurrences == 3

    def test_dedup_same_desc_diff_model_creates_new(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("same risk", model_name="gpt2")
        rr.add("same risk", model_name="gpt2-medium")  # different model → new entry
        assert len(rr) == 2

    def test_dedup_case_insensitive(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("F1 too low", model_name="m")
        rr.add("f1 too low", model_name="m")   # same when stripped/lowered
        assert len(rr) == 1

    def test_no_dedup_flag(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("same", model_name="m", deduplicate=False)
        rr.add("same", model_name="m", deduplicate=False)
        assert len(rr) == 2

    def test_severity_values(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        for sev in ("critical", "high", "medium", "low", "info"):
            rr.add(f"risk_{sev}", model_name="m", severity=sev, deduplicate=False)
        assert len(rr) == 5

    def test_multiple_models_tracked(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        for m in ("gpt2", "gpt2-medium", "gpt2-large"):
            rr.add(f"risk for {m}", model_name=m)
        assert len(rr) == 3


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — queries
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskRegisterQuery:

    @pytest.fixture
    def populated(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("critical open",   model_name="m1", severity="critical")
        rr.add("high open",       model_name="m1", severity="high")
        rr.add("medium open",     model_name="m1", severity="medium")
        rr.add("low open",        model_name="m2", severity="low")
        rr.add("info open",       model_name="m2", severity="info")
        # mitigate one
        e = rr.by_severity("high")[0]
        rr.set_status(e.risk_id, "mitigated")
        return rr

    def test_all_risks_sorted_by_severity(self, populated):
        risks = populated.all_risks()
        order = [r.severity for r in risks]
        # critical should come before high, high before medium, etc.
        idx = {s: i for i, s in enumerate(("critical","high","medium","low","info"))}
        for a, b in zip(order, order[1:]):
            assert idx.get(a, 99) <= idx.get(b, 99)

    def test_open_risks_excludes_mitigated(self, populated):
        open_r = populated.open_risks()
        assert all(r.status == "open" for r in open_r)
        assert len(open_r) == 4   # 5 total, 1 mitigated

    def test_by_severity_filter(self, populated):
        crit = populated.by_severity("critical")
        assert len(crit) == 1
        assert crit[0].severity == "critical"

    def test_by_model_filter(self, populated):
        m2_risks = populated.by_model("m2")
        assert len(m2_risks) == 2
        assert all(r.model_name == "m2" for r in m2_risks)

    def test_by_status_filter(self, populated):
        mit = populated.by_status("mitigated")
        assert len(mit) == 1

    def test_get_by_id(self, populated):
        e = populated.all_risks()[0]
        fetched = populated.get(e.risk_id)
        assert fetched is not None
        assert fetched.risk_id == e.risk_id

    def test_get_missing_id_returns_none(self, populated):
        assert populated.get("nonexistent-id-xyz") is None


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — set_status
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskRegisterSetStatus:

    def test_set_status_mitigated(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("risk", model_name="m")
        rr.set_status(e.risk_id, "mitigated")
        assert rr.get(e.risk_id).status == "mitigated"

    def test_set_status_accepted(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("risk", model_name="m")
        rr.set_status(e.risk_id, "accepted", notes="Accepted by CISO")
        entry = rr.get(e.risk_id)
        assert entry.status == "accepted"
        assert "CISO" in entry.notes

    def test_set_status_escalated(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("risk", model_name="m")
        rr.set_status(e.risk_id, "escalated")
        assert rr.get(e.risk_id).status == "escalated"

    def test_notes_append(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("risk", model_name="m", notes="Initial note")
        rr.set_status(e.risk_id, "mitigated", notes="Mitigated by retraining")
        entry = rr.get(e.risk_id)
        assert "Initial note" in entry.notes
        assert "Mitigated by retraining" in entry.notes

    def test_set_status_invalid_id_raises(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        with pytest.raises(ValueError, match="not found"):
            rr.set_status("bad-id", "mitigated")

    def test_set_status_persists_to_disk(self, tmp_path):
        path = tmp_path / "r.json"
        rr   = RiskRegister(path)
        e    = rr.add("risk", model_name="m")
        rr.set_status(e.risk_id, "mitigated", notes="done")

        rr2 = RiskRegister(path)
        assert rr2.get(e.risk_id).status == "mitigated"


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — remove
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskRegisterRemove:

    def test_remove_entry(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("risk", model_name="m")
        rr.remove(e.risk_id)
        assert len(rr) == 0

    def test_remove_nonexistent_is_silent(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.remove("does-not-exist")   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — trend_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendSummary:

    def test_empty_register_summary(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        s  = rr.trend_summary()
        assert s["total"]             == 0
        assert s["open"]              == 0
        assert s["compliance_health"] == "green"

    def test_health_red_on_critical(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("critical issue", model_name="m", severity="critical")
        assert rr.trend_summary()["compliance_health"] == "red"

    def test_health_amber_on_high_no_critical(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("high issue", model_name="m", severity="high")
        assert rr.trend_summary()["compliance_health"] == "amber"

    def test_health_green_when_all_mitigated(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e  = rr.add("critical", model_name="m", severity="critical")
        rr.set_status(e.risk_id, "mitigated")
        assert rr.trend_summary()["compliance_health"] == "green"

    def test_summary_counts(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        e1 = rr.add("a", model_name="m", severity="high")
        e2 = rr.add("b", model_name="m", severity="medium")
        e3 = rr.add("c", model_name="m", severity="low")
        rr.set_status(e1.risk_id, "mitigated")
        rr.set_status(e2.risk_id, "accepted")

        s = rr.trend_summary()
        assert s["total"]     == 3
        assert s["open"]      == 1
        assert s["mitigated"] == 1
        assert s["accepted"]  == 1

    def test_by_model_counts(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("r1", model_name="gpt2")
        rr.add("r2", model_name="gpt2")
        rr.add("r3", model_name="gpt2-medium")
        s = rr.trend_summary()
        assert s["by_model"]["gpt2"]       == 2
        assert s["by_model"]["gpt2-medium"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — to_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestToMarkdown:

    def test_empty_register_markdown(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        md = rr.to_markdown()
        assert "_No risks recorded._" in md

    def test_markdown_contains_table_headers(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("test risk", model_name="gpt2", severity="high")
        md = rr.to_markdown()
        assert "| ID |" in md
        assert "Severity" in md
        assert "Status"   in md
        assert "Model"    in md
        assert "Article"  in md

    def test_markdown_shows_compliance_health(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("critical risk", model_name="m", severity="critical")
        md = rr.to_markdown()
        assert "compliance_health" in md.lower() or "RED" in md.upper()

    def test_markdown_truncates_long_description(self, tmp_path):
        rr  = RiskRegister(tmp_path / "r.json")
        rr.add("x" * 200, model_name="m")
        md  = rr.to_markdown()
        # should be truncated with ellipsis
        assert "…" in md


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — to_json
# ─────────────────────────────────────────────────────────────────────────────

class TestToJson:

    def test_json_schema(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("risk", model_name="m", severity="high")
        j  = json.loads(rr.to_json())
        assert j["schema_version"]  == "1.0"
        assert "regulation"         in j
        assert "generated_at"       in j
        assert "summary"            in j
        assert "risks"              in j
        assert len(j["risks"])      == 1

    def test_json_risk_fields(self, tmp_path):
        rr = RiskRegister(tmp_path / "r.json")
        rr.add("test", model_name="gpt2", article="Article 13", severity="critical")
        j    = json.loads(rr.to_json())
        risk = j["risks"][0]
        assert risk["description"] == "test"
        assert risk["model_name"]  == "gpt2"
        assert risk["article"]     == "Article 13"
        assert risk["severity"]    == "critical"
        assert risk["status"]      == "open"


# ─────────────────────────────────────────────────────────────────────────────
# RiskRegister — ingest_annex_report (offline mock)
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestAnnexReport:

    class _MockAnnex:
        """Minimal mock that returns a to_json() with faithfulness risk + identified risks."""
        def __init__(self, f1=0.28, risks=None, model="gpt2"):
            self._f1     = f1
            self._risks  = risks or []
            self._model  = model

        def to_json(self):
            return json.dumps({
                "compliance_status":  "non_compliant",
                "risk_classification": "other_high_risk",
                "generated_at":        "2026-03-20T10:00:00+00:00",
                "sections": {
                    "1_general_description": {"ai_system_name": self._model},
                    "3_monitoring_control":  {
                        "f1_score":              self._f1,
                        "explainability_grade":  "D" if self._f1 < 0.30 else "C",
                    },
                    "5_risk_management": {
                        "faithfulness_risk_flag": self._f1 < 0.50,
                        "identified_risks":       self._risks,
                    },
                },
            })

    def test_faithfulness_risk_flag_adds_entry(self, tmp_path):
        rr    = RiskRegister(tmp_path / "r.json")
        annex = self._MockAnnex(f1=0.28)
        added = rr.ingest_annex_report(annex, model_name="gpt2")
        assert len(added) >= 1
        assert any("F1" in r.description for r in added)

    def test_no_flag_when_f1_ok(self, tmp_path):
        rr    = RiskRegister(tmp_path / "r.json")
        annex = self._MockAnnex(f1=0.80)
        added = rr.ingest_annex_report(annex, model_name="gpt2")
        assert len(added) == 0

    def test_identified_risks_ingested(self, tmp_path):
        rr    = RiskRegister(tmp_path / "r.json")
        risks = [
            {"risk": "Backup mechanisms detected", "article": "Article 13", "severity": "medium"},
            {"risk": "Concentration risk: >60% attribution on one head"},
        ]
        annex = self._MockAnnex(f1=0.80, risks=risks)
        added = rr.ingest_annex_report(annex)
        assert len(added) == 2

    def test_ingest_deduplicates_on_repeated_call(self, tmp_path):
        rr    = RiskRegister(tmp_path / "r.json")
        risks = [{"risk": "Backup mechanisms"}]
        annex = self._MockAnnex(f1=0.28, risks=risks)
        rr.ingest_annex_report(annex, model_name="gpt2")
        rr.ingest_annex_report(annex, model_name="gpt2")  # second run same model
        # Both risks should deduplicate — total entries stay the same
        assert len(rr) == 2   # faithfulness + backup mechanisms (both deduped)

    def test_ingest_model_name_override(self, tmp_path):
        rr    = RiskRegister(tmp_path / "r.json")
        annex = self._MockAnnex(f1=0.28, model="from-json")
        rr.ingest_annex_report(annex, model_name="override-model")
        assert all(r.model_name == "override-model" for r in rr.all_risks())

    def test_ingest_returns_list(self, tmp_path):
        rr    = RiskRegister(tmp_path / "r.json")
        annex = self._MockAnnex(f1=0.28)
        result = rr.ingest_annex_report(annex)
        assert isinstance(result, list)
